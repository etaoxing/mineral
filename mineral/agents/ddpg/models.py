import torch
import torch.nn as nn

from ...nets import MLP, Dist
from .utils import weight_init_orthogonal_, weight_init_uniform_


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        tanh_policy=True,
        fixed_sigma=None,
        mlp_kwargs=dict(units=[512, 256, 128], act_type="ELU"),
        dist_kwargs={},
        weight_init=None,
    ):
        super().__init__()
        self.tanh_policy = tanh_policy
        self.fixed_sigma = fixed_sigma

        self.actor_mlp = MLP(state_dim, **mlp_kwargs)
        self.mu = nn.Linear(self.actor_mlp.out_dim, action_dim)
        if self.tanh_policy:
            pass
        else:
            if self.fixed_sigma is None:
                pass
            elif self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = nn.Linear(self.actor_mlp.out_dim, action_dim)
            self.dist = Dist(**dist_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init == None:
            pass
        elif self.weight_init == "orthogonal":  # drqv2
            self.apply(weight_init_orthogonal_)
        elif self.weight_init == "uniform":  # original DDPG paper
            self.apply(weight_init_uniform_)
            nn.init.uniform_(self.mu.weight, -0.003, 0.003)
        else:
            raise NotImplementedError(self.weight_init)

    def forward(self, x, std=None):
        x = self.actor_mlp(x)
        mu = self.mu(x)
        if self.tanh_policy:  # DDPG
            mu = mu.tanh()
            sigma, distr = None, None
        else:  # SAC
            if self.fixed_sigma is None:
                assert std is not None
                sigma = std
            elif self.fixed_sigma:
                sigma = self.sigma
            else:
                sigma = self.sigma(x)
            mu, sigma, distr = self.dist(mu, sigma)
        return mu, sigma, distr


class EnsembleQ(nn.Module):
    def __init__(self, state_dim, action_dim, n_critics=2, mlp_kwargs={}, weight_init=None):
        super().__init__()
        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            q = MLP(state_dim + action_dim, out_dim=1, plain_last=True, **mlp_kwargs)
            critics.append(q)
        self.critics = nn.ModuleList(critics)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        for critic in self.critics:
            if self.weight_init == None:
                pass
            elif self.weight_init == "orthogonal":  # drqv2
                critic.apply(weight_init_orthogonal_)
            elif self.weight_init == "uniform":  # original DDPG paper
                critic.apply(weight_init_uniform_)
                nn.init.uniform_(critic.mlp[-1].weight, -0.003, 0.003)
            else:
                raise NotImplementedError(self.weight_init)

    def forward(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        Qs = [critic(input_x) for critic in self.critics]
        return Qs

    def get_q_min(self, state, action):
        Qs = self.forward(state, action)
        return torch.min(torch.stack(Qs), dim=0).values

    def get_q_values(self, state, action):
        return self.forward(state, action)


class DistributionalEnsembleQ(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        v_min=-10,
        v_max=10,
        num_atoms=51,
        n_critics=2,
        mlp_kwargs={},
        weight_init=None,
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms)

        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            q = MLP(state_dim + action_dim, out_dim=num_atoms, plain_last=True, **mlp_kwargs)
            critics.append(q)
        self.critics = nn.ModuleList(critics)

        self.weight_init = weight_init
        self.reset_parameters()

    @property
    def distl(self):
        return True

    def reset_parameters(self):
        if self.weight_init != None:
            raise NotImplementedError(self.weight_init)

    def forward(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        Qs = [critic(input_x) for critic in self.critics]
        return Qs

    def get_q_min(self, state, action):
        Qs = self.get_q_values(state, action)
        Qs = [torch.sum(Q * self.z_atoms.to(Q.device), dim=1) for Q in Qs]
        return torch.min(torch.stack(Qs), dim=0).values

    def get_q_values(self, state, action):
        Qs = self.forward(state, action)
        Qs = [torch.softmax(Q, dim=1) for Q in Qs]
        return Qs
