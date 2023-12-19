from typing import List

import torch
import torch.nn as nn


def MLP(
    units: List[int],
    plain_last: bool = False,
    norm_type="BatchNorm1d",
    act_type="ReLU",
):
    layers = []

    in_size = units[0]
    for i in range(1, len(units)):
        out_size = units[i]
        l = nn.Linear(in_size, out_size)
        layers.append(l)

        if plain_last and i == len(units) - 1:
            break

        if norm_type is not None:
            module = torch.nn
            Cls = getattr(module, norm_type)
            layers.append(Cls(out_size))

        if act_type is not None:
            module = torch.nn.modules.activation
            Cls = getattr(module, act_type)
            layers.append(Cls())

        in_size = out_size
    return nn.Sequential(*layers)


class STNkd(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_units=[64, 128, 1024],
        mlp_units=[1024, 512, 256],
        norm_type="BatchNorm1d",
        act_type="ReLU",
    ):
        super().__init__()
        # using linear layers for conv1d
        self.conv = MLP([input_dim] + conv_units, norm_type=norm_type, act_type=act_type)
        self.mlp = MLP(mlp_units + [input_dim**2], plain_last=True, norm_type=norm_type, act_type=act_type)

        self.register_buffer("identity", torch.eye(input_dim))

    def forward(self, x):
        B, N, D = x.shape

        x = x.reshape(B * N, D)
        x = self.conv(x)
        x = x.view(B, N, -1)

        x, _ = torch.max(x, 1)

        x = self.mlp(x)
        x = x.view(B, D, D)

        x = x + self.identity[None]
        return x


class PointNet(nn.Module):
    def __init__(
        self,
        pcd_shapes,
        node_feature_dim=0,
        global_feature_dim=1024,
        local_feature_dim=None,
        feature_units=[64, 128],
        stn_kwargs=dict(conv_units=[64, 128, 1024], mlp_units=[1024, 512, 256]),
        fstn_kwargs={},
        feature_transform=False,
        norm_type="BatchNorm1d",
        act_type="ReLU",
        plain_last=False,  # different than orig impl. that has bn but no relu in the last layer
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.global_feature_dim = global_feature_dim
        self.feature_transform = feature_transform
        self.feature_units = feature_units

        if local_feature_dim is not None:
            raise ValueError
        self.local_feature_dim = feature_units[0]

        D = 3 + node_feature_dim
        norm_act_kwargs = dict(norm_type=norm_type, act_type=act_type)
        self.stn = STNkd(D, **dict(**norm_act_kwargs, **stn_kwargs))
        if self.feature_transform:
            self.fstn = STNkd(feature_units[0], **dict(**norm_act_kwargs, **fstn_kwargs))

        self.feature_l0 = MLP([D, feature_units[0]], **norm_act_kwargs)
        self.feature_l1 = MLP(feature_units + [global_feature_dim], **norm_act_kwargs, plain_last=plain_last)

    def forward(self, data):
        x, pos = data
        B, N, _ = pos.shape
        if x is not None:
            x = torch.cat([pos, x], -1)
        else:
            x = pos

        trans = self.stn(x)
        x = torch.bmm(x, trans)
        x = x.view(B * N, -1)
        x = self.feature_l0(x)
        if self.feature_transform:
            x = x.view(B, N, -1)
            feature_trans = self.fstn(x)
            x = torch.bmm(x, feature_trans)
            local_x = x  # (B, N, D)
            x = x.view(B * N, -1)
        else:
            local_x = x
        x = self.feature_l1(x)
        x = x.view(B, N, -1)

        global_x, _ = torch.max(x, 1)
        # local_x = torch.cat([global_x.unsqueeze(1).repeat(1, local_x.shape[1], 1), local_x], dim=-1)
        return global_x, local_x
