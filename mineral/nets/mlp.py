import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        units=[512, 256, 128],
        norm_type=None,
        norm_kwargs={},
        act_type="ReLU",
        act_kwargs=dict(inplace=True),
        plain_last=False,
    ):
        super().__init__()
        if out_dim is not None:
            units = [*units, out_dim]
        self.in_dim = in_dim
        self.out_dim = units[-1]
        self.units = units

        in_size = in_dim
        layers = []
        for i, out_size in enumerate(units):
            lin = nn.Linear(in_size, out_size)
            layers.append(lin)
            if plain_last and i == len(units) - 1:
                break
            if norm_type is not None:
                module = torch.nn
                Cls = getattr(module, norm_type)
                norm = Cls(out_size, **norm_kwargs)
                layers.append(norm)
            if act_type is not None:
                module = torch.nn.modules.activation
                Cls = getattr(module, act_type)
                act = Cls(**act_kwargs)
                layers.append(act)
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
