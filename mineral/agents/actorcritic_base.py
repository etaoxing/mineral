import collections
import os
import re

import numpy as np
import torch

from ..common.metrics_tracker import MetricsTracker


class ActorCriticBase:
    def __init__(self, env, output_dir, full_cfg, accelerator=None, datasets=None):
        self.output_dir = output_dir
        self.full_cfg = full_cfg

        self.rank = -1
        self.device = full_cfg.rl_device
        self.multi_gpu = full_cfg.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))

            assert accelerator is not None
            self.accelerator = accelerator
            self.device = self.accelerator.device

        # ---- Datasets ----
        self.datasets = datasets

        # ---- Environment ----
        self.env = env
        action_space = self.env.action_space
        self.action_dim = action_space.shape[0]
        self.env_autoresets = full_cfg.task.get('env_autoresets', True)  # set to False to explicitly call env.reset

        # ---- Inputs ----
        self.obs_keys_cpu = re.compile(full_cfg.agent.get('obs_keys_cpu', '$^'))
        self.input_keys_normalize = re.compile(full_cfg.agent.get('input_keys_normalize', ''))
        self.normalize_input = full_cfg.agent.get('normalize_input', False)
        self.observation_space = self.env.observation_space
        try:
            obs_space = {k: v.shape for k, v in self.observation_space.spaces.items()}
        except:
            obs_space = {'obs': self.observation_space.shape}
        self.obs_space = obs_space

        # ---- Logging ----
        self.metrics_tracker = MetricsTracker(full_cfg, self.output_dir, self.num_actors, self.device)

        self.ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.print_every = full_cfg.agent.get('print_every', -1)
        self.ckpt_every = full_cfg.agent.get('ckpt_every', -1)
        self.eval_every = full_cfg.agent.get('eval_every', -1)

        self.epoch = -1
        self.mini_epoch = -1
        self.agent_steps = 0

    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        raise NotImplementedError

    def get_actions(self, obs, sample: bool = True):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def save(self, f):
        raise NotImplementedError

    def load(self, f):
        raise NotImplementedError

    def _convert_obs(self, obs):
        if not isinstance(obs, dict):
            obs = {'obs': obs}

        # Copy obs dict since env.step may modify it (ie. IsaacGymEnvs)
        _obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                _obs[k] = torch.tensor(v, device=self.device if not re.match(self.obs_keys_cpu, k) else 'cpu')
            else:
                # assert isinstance(v, torch.Tensor)
                _obs[k] = v
        return _obs

    @staticmethod
    def _handle_timeout(dones, info, timeout_keys=('time_outs', 'TimeLimit.truncated')):
        timeout_envs = None
        for timeout_key in timeout_keys:
            if timeout_key in info:
                timeout_envs = info[timeout_key]
                break
        if timeout_envs is not None:
            dones = dones * (~timeout_envs)
        return dones
