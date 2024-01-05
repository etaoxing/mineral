import numpy as np
import torch
import torch.nn as nn

from ...nets import MLP, Dist, MultiEncoder


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        fan_out = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.41421356237)  # np.sqrt(2)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space,
        action_dim,
        mlp_kwargs=dict(units=[512, 256, 128], act_type="ELU"),
        separate_value_mlp=True,
        fixed_sigma=True,
        actor_dist_kwargs=dict(dist_type='normal'),
        encoder=None,
        encoder_kwargs=None,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.separate_value_mlp = separate_value_mlp
        self.fixed_sigma = fixed_sigma

        if 'obs' in obs_space:
            self.encoder = nn.Identity()
            mlp_in_dim = obs_space['obs'][0]
        else:
            self.encoder = MultiEncoder(obs_space, encoder_kwargs)
            mlp_in_dim = self.encoder.out_dim

        self.actor_mlp = MLP(mlp_in_dim, **mlp_kwargs)
        if self.separate_value_mlp:
            self.value_mlp = MLP(mlp_in_dim, **mlp_kwargs)
        out_size = self.actor_mlp.out_dim
        self.value = nn.Linear(out_size, 1)
        self.mu = nn.Linear(out_size, action_dim)
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = nn.Linear(out_size, action_dim)
        self.dist = Dist(**actor_dist_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.actor_mlp.apply(weight_init)
        if self.separate_value_mlp:
            self.value_mlp.apply(weight_init)
        self.value.apply(weight_init)
        self.mu.apply(weight_init)

        # value output layer with scale 1
        # policy output layer with scale 0.01
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        if self.fixed_sigma:
            nn.init.constant_(self.sigma, 0)
        else:
            nn.init.orthogonal_(self.sigma.weight, gain=0.01)
            nn.init.zeros_(self.sigma.bias)

    def forward(self, input_dict, **kwargs):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict, **kwargs)
        mu, sigma, distr = self.dist(mu, logstd)
        entropy = distr.entropy().sum(dim=-1)
        result = {
            'values': value,
            'entropy': entropy,
            'mu': mu,
            'sigma': sigma,
            'distr': distr,
        }
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
            result['prev_neglogp'] = torch.squeeze(prev_neglogp)
        return result

    @torch.no_grad()
    def act(self, obs_dict, sample=True, **kwargs):
        mu, logstd, value = self._actor_critic(obs_dict, **kwargs)
        if not sample:
            return mu
        mu, sigma, distr = self.dist(mu, logstd)
        selected_action = distr.sample()
        neglogp = -distr.log_prob(selected_action).sum(1)
        result = {
            'neglogp': neglogp,
            'values': value,
            'actions': selected_action,
            'mu': mu,
            'sigma': sigma,
            'distr': distr,
        }
        return result

    def _encode(self, obs_dict):
        if 'obs' in self.obs_space:
            z = obs_dict['obs']
        else:
            encoder_out = self.encoder(obs_dict)
            z = encoder_out['z']
        return z

    def _actor_critic(self, obs_dict):
        z = self._encode(obs_dict)

        x = self.actor_mlp(z)
        mu = self.mu(x)

        if self.separate_value_mlp:
            x = self.value_mlp(z)
        value = self.value(x)

        if self.fixed_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma(x)
        return mu, mu * 0 + sigma, value
