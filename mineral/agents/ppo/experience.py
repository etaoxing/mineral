import re

import torch
from torch.utils.data import Dataset


def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(self, num_envs, horizon_len, batch_size, minibatch_size, obs_space, act_dim, device, cpu_obs_keys):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_len
        self.cpu_obs_keys = cpu_obs_keys

        self.data_dict = None
        self.obs_space = obs_space
        self.act_dim = act_dim
        self.storage_dict = {
            'rewards': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            'values': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            'neglogp': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.float32, device=self.device),
            'dones': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.uint8, device=self.device),
            'actions': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device
            ),
            'mu': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device
            ),
            'sigma': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device
            ),
            'returns': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
        }

        obs_dict = {}
        for k, v in self.obs_space.items():
            buffer = torch.zeros(
                (self.transitions_per_env, self.num_envs, *v),
                dtype=torch.float32,
                device='cpu' if re.match(self.cpu_obs_keys, k) else self.device,
            )
            obs_dict[k] = buffer
        self.storage_dict['obses'] = obs_dict

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        return (
            input_dict['values'],
            input_dict['neglogp'],
            input_dict['advantages'],
            input_dict['mu'],
            input_dict['sigma'],
            input_dict['returns'],
            input_dict['actions'],
            input_dict['obses'],
        )

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mu'][start:end] = mu
        self.data_dict['sigma'][start:end] = sigma

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index, :] = v
        else:
            self.storage_dict[name][index, :] = val

    def compute_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict['rewards'][t] + gamma * next_values * next_nonterminal - self.storage_dict['values'][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            if isinstance(v, dict):
                self.data_dict[k] = {}
                for kd, vd in v.items():
                    self.data_dict[k][kd] = transform_op(vd)
            else:
                self.data_dict[k] = transform_op(v)
        advantages = self.data_dict['returns'] - self.data_dict['values']
        self.data_dict['advantages'] = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict
