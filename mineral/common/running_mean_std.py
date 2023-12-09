import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
