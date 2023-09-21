import collections
import os
import re

import numpy as np
import torch
from omegaconf import OmegaConf

from ..common.tracker import Tracker
from ..common.writer import TensorboardWriter, WandbWriter


class ActorCriticBase:
    def __init__(self, env, output_dir, full_cfg):
        self.env = env
        self.output_dir = output_dir
        self.full_cfg = full_cfg

        self.rank = -1
        self.device = full_cfg.rl_device

        self.make_writers(full_cfg)

    @property
    def step(self):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def play_steps(self):
        raise NotImplementedError

    def make_writers(self, full_cfg):
        # ---- Logging ----
        self.env_render = full_cfg.env_render
        info_keys_cfg = full_cfg.agent.get('info_keys', {})
        self.info_keys_video = info_keys_cfg.get('video', '$^')
        self.info_keys_sum = info_keys_cfg.get('sum', '$^')
        self.info_keys_min = info_keys_cfg.get('min', '$^')
        self.info_keys_max = info_keys_cfg.get('max', '$^')
        self.info_keys_final = info_keys_cfg.get('final', '$^')
        self.info_keys_scalar = info_keys_cfg.get('scalar', '$^')
        self.save_video_every = full_cfg.agent.get('save_video_every', 0)
        self.save_video_consecutive = full_cfg.agent.get('save_video_consecutive', 0)

        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        self.tb_dir = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # --- Tracking ---
        self._episode_info = {}
        self._video_buf = collections.defaultdict(list)

        self.current_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)

        self.episode_rewards = Tracker(100)
        self.episode_lengths = Tracker(100)

        # ---- Wandb / Tensorboard Logger ----
        self._info_extra = {}
        self._info_video = None
        self._info_keys_stats = collections.defaultdict(list)
        resolved_config = OmegaConf.to_container(full_cfg, resolve=True)
        self._writers = [
            WandbWriter(),
            TensorboardWriter(self.tb_dir, resolved_config),
        ]

    def write_metrics(self, step, metrics):
        for k, v in self._info_keys_stats.items():
            v = np.concatenate(v, 0)
            metrics[f'episode/{k}'] = np.nanmean(v).item()
        self._info_keys_stats.clear()

        for k, v in self._info_extra.items():
            metrics[f'extras/{k}'] = v

        if self._info_video is not None:
            metrics.update(self._info_video)
            self._info_video = None

        summary = tuple([(step, k, v) for k, v in metrics.items()])
        [w(summary) for w in self._writers]

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

    def _update_tracker(self, rewards, done_indices, infos, save_video=False):
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

        if save_video:
            if self.env_render:
                v = self.env.render(mode='rgb_array')
                v = self._reshape_env_render('env_render', v)
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                self._video_buf['env_render'].append(v)

            for k, v in self.obs.items():
                if re.match(self.info_keys_video, k):
                    v = self._reshape_env_render(k, v)
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    self._video_buf[k].append(v)

        ep = self._episode_info
        for k, v in self.obs.items():
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
