import numpy as np
import torch



def grad_norm(params):
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad**2)
    return torch.sqrt(grad_norm)


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle=False, drop_last=False):
        self.obs = {k: v.view(-1, v.shape[-1]) for k, v in obs.items()}
        self.target_values = target_values.view(-1)
        self.N = self.target_values.shape[0]
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.N // self.batch_size
        else:
            self.length = ((self.N - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.N)
        self.obs = {k: v[index, :] for k, v in self.obs.items()}
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.N)

        obs = {k: v[start_idx:end_idx, :] for k, v in self.obs.items()}
        target_values = self.target_values[start_idx:end_idx]
        return obs, target_values
