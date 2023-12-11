import math

import torch
import torch.nn as nn


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        # tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))
        tar.mul_(1.0 - tau).add_(cur * tau)


def weight_init_orthogonal_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')  # 1.41421356237
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def weight_init_uniform_(m):
    if isinstance(m, nn.Linear):
        variance_initializer_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
# this comes from SoRB, implemented in tf
def variance_initializer_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        scale /= max(1.0, fan_in)
    elif mode == 'fan_out':
        scale /= max(1.0, fan_out)
    else:
        raise ValueError(mode)

    if distribution == 'uniform':
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError(distribution)
