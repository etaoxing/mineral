import collections
import os
import re

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .tracker import Tracker
from .writer import TensorboardWriter, WandbWriter


class MetricsTracker(nn.Module):
    def __init__(self, full_cfg, output_dir, num_actors, device):
        super().__init__()
        self.output_dir = output_dir
        self.num_actors = num_actors
        self.device = device

        self.make_writers(full_cfg)

    def make_writers(self, full_cfg):
        # ---- Logging ----
        self.env_render = full_cfg.env_render
        info_keys_cfg = full_cfg.agent.get('info_keys', {})
        self.info_keys_video = re.compile(info_keys_cfg.get('video', '$^'))
        self.info_keys_sum = re.compile(info_keys_cfg.get('sum', '$^'))
        self.info_keys_min = re.compile(info_keys_cfg.get('min', '$^'))
        self.info_keys_max = re.compile(info_keys_cfg.get('max', '$^'))
        self.info_keys_final = re.compile(info_keys_cfg.get('final', '$^'))
        self.info_keys_scalar = re.compile(info_keys_cfg.get('scalar', '$^'))
        self.save_video_every = full_cfg.agent.get('save_video_every', 0)
        self.save_video_consecutive = full_cfg.agent.get('save_video_consecutive', 0)

        # ---- Tensorboard Dir ----
        self.tb_dir = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.tb_dir, exist_ok=True)

        # --- Tracking ---
        self._episode_info = {}
        self._video_buf = collections.defaultdict(list)

        self.current_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)

        tracker_len = full_cfg.agent.get('tracker_len', 100)
        self.episode_rewards = Tracker(tracker_len)
        self.episode_lengths = Tracker(tracker_len)

        # ---- Wandb / Tensorboard Logger ----
        self._info_extra = {}
        self._info_video = None
        self._info_keys_stats = collections.defaultdict(list)
        resolved_config = OmegaConf.to_container(full_cfg, resolve=True)
        self._writers = [
            WandbWriter(),
            TensorboardWriter(self.tb_dir, resolved_config),
        ]

    @staticmethod
    def _reshape_env_render(k, v):
        if len(v.shape) == 3:  # H, W, C
            v = v[None][None]  # -> B, T, H, W, C
        elif len(v.shape) == 4:  # B, H, W, C
            v = v[:, None, ...]  # -> B, T, H, W, C
        elif len(v.shape) == 5:  # B, T, H, W, C
            pass
        else:
            raise RuntimeError(f'Unsupported {k} shape {v.shape}')
        return v

    def update_tracker(self, epoch, env, obs, rewards, done_indices, infos):
        self.current_rewards += rewards
        self.current_lengths += 1
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        self.current_rewards[done_indices] = 0
        self.current_lengths[done_indices] = 0

        for k, v in infos.items():
            if re.match(self.info_keys_scalar, k):
                # assert isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                self._info_extra[k] = v.item()

        save_video = (self.save_video_every > 0) and (epoch % self.save_video_every < self.save_video_consecutive)
        if save_video:
            if self.env_render:
                v = env.render(mode='rgb_array')
                v = self._reshape_env_render('env_render', v)
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                self._video_buf['env_render'].append(v)

            for k, v in obs.items():
                if re.match(self.info_keys_video, k):
                    v = self._reshape_env_render(k, v)
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    self._video_buf[k].append(v)

        ep = self._episode_info
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            if re.match(self.info_keys_sum, k):
                _k = f'{k}_sum'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] += v
            if re.match(self.info_keys_min, k):
                _k = f'{k}_min'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] = torch.fmin(ep[_k], v)
            if re.match(self.info_keys_max, k):
                _k = f'{k}_max'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] = torch.fmax(ep[_k], v)
            if re.match(self.info_keys_final, k):
                _k = f'{k}_final'
                if _k not in ep:
                    ep[_k] = torch.zeros_like(v)
                ep[_k][done_indices] = v[done_indices]

        if len(done_indices) > 0:
            for k in ep.keys():
                v = ep[k][done_indices].cpu().numpy()
                self._info_keys_stats[k].append(v)
                if 'sum' in k or 'final' in k:
                    ep[k][done_indices] = 0
                else:
                    ep[k][done_indices] = torch.nan

    def flush_video_buf(self, epoch):
        if self.save_video_every > 0:
            # saved video steps depends on horizon_length in play_steps()
            if (epoch % self.save_video_every) == (self.save_video_consecutive - 1):
                self._info_video = {f'video/{k}': np.concatenate(v, 1) for k, v in self._video_buf.items()}
                self._video_buf = collections.defaultdict(list)

    def write_metrics(self, step, metrics):
        for k, v in self._info_keys_stats.items():
            v = np.concatenate(v, 0)
            metrics[f'episode/{k}'] = np.nanmean(v).item()
            metrics[f'episode_stds/{k}'] = np.nanstd(v).item()
        self._info_keys_stats.clear()

        for k, v in self._info_extra.items():
            metrics[f'extras/{k}'] = v

        if self._info_video is not None:
            metrics.update(self._info_video)
            self._info_video = None

        summary = tuple([(step, k, v) for k, v in metrics.items()])
        [w(summary) for w in self._writers]
