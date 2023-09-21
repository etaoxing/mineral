import re

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer('running_mean', torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(in_size, dtype=torch.float64))
        self.register_buffer('count', torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        # get output
        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

    def __repr__(self):
        return f'RunningMeanStd({self.insize})'


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
