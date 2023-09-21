import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


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
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultiEncoder(nn.Module):
    def __init__(self, obs_space, config):
        super().__init__()
        cnn_keys = config.encoder.get('cnn_keys', '$^')
        pcd_keys = config.encoder.get('pcd_keys', '$^')
        mlp_keys = config.encoder.get('mlp_keys', '$^')

        excluded = {}
        shapes = {k: v for k, v in obs_space.items() if k not in excluded and not k.startswith('info_')}
        print('Encoder input shapes:', shapes)
        self.cnn_shapes = {k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))}
        self.pcd_shapes = {k: v for k, v in shapes.items() if (len(v) == 2 and re.match(pcd_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.pcd_shapes, **self.mlp_shapes}
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder PCD shapes:', self.pcd_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)

        self.out_dim = 0
        self.out_dim_local = None
        if self.cnn_shapes:
            cnn, cnn_kwargs = config.encoder.cnn, config.encoder.cnn_kwargs

            if cnn == 'resnet':
                raise NotImplementedError
            else:
                raise NotImplementedError(cnn)

        if self.pcd_shapes:
            pcd, pcd_kwargs = config.encoder.pcd, config.encoder.pcd_kwargs

            if pcd == 'pointnet2':
                raise NotImplementedError
            else:
                raise NotImplementedError(pcd)
        if self.mlp_shapes:
            tensor_shape = sum([np.prod(v, dtype=int) for v in self.mlp_shapes.values()]).item()
            self.out_dim += tensor_shape
        print('Encoder out_dim:', self.out_dim)

    def cnn(self, x):
        inputs = torch.cat([x[k] for k in self.cnn_shapes], -1)
        output = self._cnn(inputs)
        output = output.reshape((output.shape[0], -1))
        return output

    def pcd(self, x):
        _pos = torch.cat([x[k] for k in self.pcd_shapes], -2)
        _x = None
        _batch = torch.arange(_pos.shape[0], device=_pos.device).repeat_interleave(_pos.shape[1])
        B, D = _pos.shape[:2]
        _pos = _pos.reshape(-1, *_pos.shape[2:])
        data = dict(x=_x, pos=_pos, batch=_batch)

        inputs = self._pcd_transforms(data)
        global_z, local_z = self._pcd(inputs)
        del inputs
        local_z = local_z.reshape(B, D, -1)
        return global_z, local_z

    def forward(self, x):
        outputs = {}
        outputs_local = None
        if self.cnn_shapes:
            outputs['cnn'] = self.cnn(x)

        if self.pcd_shapes:
            global_z, local_z = self.pcd(x)
            outputs['pcd'] = global_z
            outputs_local = local_z

        if self.mlp_shapes:
            inputs = [x[k][..., None] if len(self.shapes[k]) == 0 else x[k] for k in self.mlp_shapes]
            inputs = torch.cat([x.reshape(x.shape[0], -1) for x in inputs], -1)
            outputs['mlp'] = inputs

        outputs = torch.cat(list(outputs.values()), -1)
        return outputs, outputs_local


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
        elif self._dist == 'normal_dv3':
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


class ActorCritic(nn.Module):
    def __init__(self, obs_space, actions_dim, config):
        super().__init__()
        self.separate_value_mlp = config.separate_value_mlp
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
        self.sigma = nn.Parameter(
            torch.zeros(actions_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)

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
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.sigma, 0)

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

        sigma = self.sigma
        return mu, mu * 0 + sigma, value

