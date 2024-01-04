import torch
import torch.nn as nn

from ...nets import MLP, Dist


def weight_init_orthogonal_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1.4142135623730951)  # sqrt(2)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)


def weight_init_(module, weight_init):
    if weight_init == None:
        pass
    elif weight_init == "orthogonal":
        module.apply(weight_init_orthogonal_)
    else:
        raise NotImplementedError(weight_init)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        fixed_sigma=True,
        init_sigma=-1.0,
        mlp_kwargs=dict(norm_type="LayerNorm", act_type="ELU"),
        dist_kwargs=dict(dist="normal"),
        weight_init="orthogonal",
    ):
        super().__init__()
        self.fixed_sigma = fixed_sigma
        self.init_sigma = init_sigma

        self.actor_mlp = MLP(state_dim, **mlp_kwargs)
        self.mu = nn.Linear(self.actor_mlp.out_dim, action_dim)
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.ones(action_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = nn.Linear(self.actor_mlp.out_dim, action_dim)
        self.dist = Dist(**dist_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        weight_init_(self, self.weight_init)

        if self.fixed_sigma:
            nn.init.constant_(self.sigma, self.init_sigma)

    def forward(self, x):
        x = self.actor_mlp(x)
        mu = self.mu(x)
        if self.fixed_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma(x)
        mu, sigma, distr = self.dist(mu, sigma)
        return mu, sigma, distr


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        mlp_kwargs=dict(act_type="ELU", norm_type="LayerNorm"),
        weight_init="orthogonal",
    ):
        super().__init__()
        self.critic_mlp = MLP(state_dim, out_dim=1, plain_last=True, **mlp_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        weight_init_(self, self.weight_init)

    def forward(self, x):
        x = self.critic_mlp(x)
        return x
