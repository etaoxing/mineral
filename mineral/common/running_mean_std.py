import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape, eps=1e-4, with_clamp=False, clamp_range=(-5.0, 5.0), initial_count='eps', dtype=torch.float32):
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.with_clamp = with_clamp
        self.clamp_range = clamp_range

        if initial_count == 'eps':
            initial_count = eps

        self.register_buffer('running_mean', torch.zeros(shape, dtype=dtype))
        self.register_buffer('running_var', torch.ones(shape, dtype=dtype))
        self.register_buffer('running_count', torch.tensor(initial_count, dtype=dtype))

    @staticmethod
    def _update_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.running_mean, self.running_var, self.running_count = self._update_from_moments(
            self.running_mean, self.running_var, self.running_count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x):
        x = (x - self.running_mean.float()) / torch.sqrt(self.running_var.float() + self.eps)
        if self.with_clamp:
            x = torch.clamp(x, min=self.clamp_range[0], max=self.clamp_range[1])
        return x

    def unnormalize(self, x):
        if self.with_clamp:
            x = torch.clamp(x, min=self.clamp_range[0], max=self.clamp_range[1])
        x = torch.sqrt(self.running_var.float() + self.eps) * x + self.running_mean.float()
        return x

    def __repr__(self):
        return f'RunningMeanStd(shape={self.shape})'
