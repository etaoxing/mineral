import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def model_utils_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def model_utils_get_activation_func(activation_name):
    if activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError('Actication func {} not defined'.format(activation_name))


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, activation, logstd=-1.0, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + hidden_dims + [action_dim]

        init_ = lambda m: model_utils_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils_get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(model_utils_get_activation_func('identity'))

        self.mu_net = nn.Sequential(*modules).to(device)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic=False):
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp()  # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()  # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)


class CriticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, activation, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + hidden_dims + [1]

        init_ = lambda m: model_utils_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils_get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)

        self.obs_dim = obs_dim

        print(self.critic)

    def forward(self, observations):
        return self.critic(observations)
