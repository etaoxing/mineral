import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..nets import MultiEncoder


class MLP(nn.Module):
    def __init__(self, input_size, units, act_type='ELU', norm_type=None):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            if norm_type is not None:
                module = torch.nn
                Cls = getattr(module, norm_type)
                layers.append(Cls(output_size))
            if act_type is not None:
                module = torch.nn.modules.activation
                Cls = getattr(module, act_type)
                layers.append(Cls())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]


class ActorCritic(nn.Module):
    def __init__(self, obs_space, actions_dim, config):
        super().__init__()
        self.separate_value_mlp = config.separate_value_mlp
        self.actor_input_ind_std = config.actor_input_ind_std
        self.act_type = config.act_type
        self.norm_type = config.norm_type
        self.units = config.mlp.units
        out_size = self.units[-1]

        self.obs_space = obs_space
        if 'obs' in obs_space:
            self.encoder = nn.Identity()
            mlp_in_dim = obs_space['obs'][0]
        else:
            self.encoder = MultiEncoder(obs_space, config)
            mlp_in_dim = self.encoder.out_dim
        self.actor_mlp = MLP(input_size=mlp_in_dim, units=self.units, act_type=self.act_type, norm_type=self.norm_type)
        if self.separate_value_mlp:
            self.value_mlp = MLP(input_size=mlp_in_dim, units=self.units, act_type=self.act_type, norm_type=self.norm_type)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_dim)

        if self.actor_input_ind_std:
            self.sigma = nn.Parameter(torch.zeros(actions_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = torch.nn.Linear(out_size, actions_dim)

        actor_dist_config = config.get('actor_dist', {})
        self.dist = Dist(**actor_dist_config)

        self.reset_parameters()

    def reset_parameters(self):
        def weight_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight.data, gain=1.41421356237)
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        self.actor_mlp.apply(weight_init)
        if self.separate_value_mlp:
            self.value_mlp.apply(weight_init)
        self.value.apply(weight_init)
        self.mu.apply(weight_init)

        # value output layer with scale 1
        # policy output layer with scale 0.01
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        if self.actor_input_ind_std:
            nn.init.constant_(self.sigma, 0)
        else:
            nn.init.orthogonal_(self.sigma.weight, gain=0.01)
            nn.init.zeros_(self.sigma.bias)

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        mu, sigma, distr = self.dist(mu, logstd)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        mu, sigma, distr = self.dist(mu, logstd)
        selected_action = distr.sample()
        neglogp = -distr.log_prob(selected_action).sum(1)
        result = {
            'neglogp': neglogp,
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict, inference=True)
        return mu

    def _encode(self, obs_dict):
        if 'obs' in self.obs_space:
            z = obs_dict['obs']
            z_local = None
        else:
            z, z_local = self.encoder(obs_dict)
        return z, z_local

    def _actor_critic(self, obs_dict, inference=False):
        z, z_local = self._encode(obs_dict)

        x = self.actor_mlp(z)
        mu = self.mu(x)

        if self.separate_value_mlp:
            x = self.value_mlp(z)
        value = self.value(x)

        if self.actor_input_ind_std:
            sigma = self.sigma
        else:
            sigma = self.sigma(x)
        return mu, mu * 0 + sigma, value


class Dist(nn.Module):
    def __init__(
        self,
        dist='normal',
        minstd=1.0,
        maxstd=1.0,
    ):
        super().__init__()
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd

    def forward(self, mu, logstd):
        if self._dist == 'normal':
            sigma = torch.exp(logstd)
            distr = D.Normal(mu, sigma)
        elif self._dist == 'normal_dreamerv3':
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * torch.sigmoid(logstd + 2.0) + lo
            mu = torch.tanh(mu)
            distr = D.Normal(mu, std)
            sigma = std
        else:
            raise NotImplementedError
        return mu, sigma, distr

    def __repr__(self):
        return f'Dist(dist={self._dist})'
