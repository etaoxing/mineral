import re

import torch
import torch.nn as nn


class PCDInputs(nn.Module):
    def __init__(self, pcd_shapes, x_keys=None, pos_keys=None, pyg_data=False):
        super().__init__()
        self.pcd_shapes = pcd_shapes
        self.x_keys = x_keys
        self.pos_keys = pos_keys
        self.pyg_data = pyg_data

        if self.x_keys is not None:
            self.x_keys = re.compile(self.x_keys)
        if self.pos_keys is not None:
            self.pos_keys = re.compile(self.pos_keys)

        if pyg_data:
            import torch_geometric.transforms as T
            from torch_geometric.data import Data
            from torch_geometric.transforms import BaseTransform

            class ToBatchTransform(BaseTransform):
                def __call__(self, data):
                    x, pos, batch = data['x'], data['pos'], data['batch']
                    if x is None:
                        return Data(pos=pos, batch=batch)
                    else:
                        return Data(x=x, pos=pos, batch=batch)

            pcd_transforms = [
                ToBatchTransform(),
            ]
            self.pcd_transforms = T.Compose(pcd_transforms)

    def get_x_and_pos(self, d):
        x, pos = None, None  # -> (node features), (xyz position)
        # TODO: support concating multiple pcds
        for k, v in d.items():
            if self.x_keys is not None and re.match(self.x_keys, k):
                x = v
            if self.pos_keys is not None and re.match(self.pos_keys, k):
                pos = v
        return x, pos

    def forward(self, d):
        x, pos = self.get_x_and_pos(d)

        if self.pyg_data:
            batch = torch.arange(pos.shape[0], device=pos.device).repeat_interleave(pos.shape[1])
            B, D = pos.shape[:2]
            pos = pos.reshape(-1, *pos.shape[2:])
            if x is not None:
                x = x.reshape(-1, *x.shape[2:])
            data = dict(x=x, pos=pos, batch=batch)

            data = self.pcd_transforms(data)
            return data
        else:
            return x, pos
