import collections
import re

import numpy as np
import torch
import torch.nn as nn

from .tracker import Tracker


class Metrics(nn.Module):
    def __init__(
        self,
        num_actors,
        device,
        env_render,
        save_video_every=0,
        save_video_consecutive=0,
        tracker_len=100,
        info_keys={},
    ):
        super().__init__()
        self.num_actors = num_actors
        self.device = device

        # --- Logging ---
        self.info_keys_sum = re.compile(info_keys.get('sum', '$^'))
        self.info_keys_min = re.compile(info_keys.get('min', '$^'))
        self.info_keys_max = re.compile(info_keys.get('max', '$^'))
        self.info_keys_final = re.compile(info_keys.get('final', '$^'))
        self.info_keys_scalar = re.compile(info_keys.get('scalar', '$^'))
        self.info_keys_video = re.compile(info_keys.get('video', '$^'))

        # --- Video ---
        self.env_render = env_render
        self.save_video_every = save_video_every
        self.save_video_consecutive = save_video_consecutive

        # --- Tracking ---
        self.current_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self._current_info = {}
        self._video_buf = collections.defaultdict(list)

        self.tracker_len = tracker_len
        self.episode_rewards = Tracker(tracker_len)
        self.episode_lengths = Tracker(tracker_len)

        self.episode_stats = collections.defaultdict(list)  # TODO: use tracker_len?
        self._info_extra = {}
        self._info_video = None

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

    def update(self, epoch, env, obs, rewards, done_indices, infos):
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

        d = self._current_info
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            if re.match(self.info_keys_sum, k):
                _k = f'{k}_sum'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] += v
            if re.match(self.info_keys_min, k):
                _k = f'{k}_min'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] = torch.fmin(d[_k], v)
            if re.match(self.info_keys_max, k):
                _k = f'{k}_max'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] = torch.fmax(d[_k], v)
            if re.match(self.info_keys_final, k):
                _k = f'{k}_final'
                if _k not in d:
                    d[_k] = torch.zeros_like(v)
                d[_k][done_indices] = v[done_indices]

        if len(done_indices) > 0:
            for k in d.keys():
                v = d[k][done_indices].cpu().numpy()
                self.episode_stats[k].append(v)

                # reset
                if 'sum' in k or 'final' in k:
                    d[k][done_indices] = 0
                else:
                    d[k][done_indices] = torch.nan

    def flush_video_buf(self, epoch):
        if self.save_video_every > 0:
            # saved video steps depends on horizon_len in play_steps()
            if (epoch % self.save_video_every) == (self.save_video_consecutive - 1):
                self._info_video = {f'video/{k}': np.concatenate(v, 1) for k, v in self._video_buf.items()}
                self._video_buf = collections.defaultdict(list)

    def result(self, metrics):
        for k, v in self.episode_stats.items():
            v = np.concatenate(v, 0)
            metrics[f'episode/{k}'] = np.nanmean(v).item()
            metrics[f'episode_stds/{k}'] = np.nanstd(v).item()
        self.episode_stats.clear()

        for k, v in self._info_extra.items():
            metrics[f'extras/{k}'] = v
        self._info_extra.clear()

        if self._info_video is not None:
            metrics.update(self._info_video)
            self._info_video = None

        return metrics
