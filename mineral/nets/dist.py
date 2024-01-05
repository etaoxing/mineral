import torch
import torch.distributions as D
import torch.nn as nn

from .distributions import SquashedNormal


class Dist(nn.Module):
    def __init__(
        self,
        dist_type='normal',
        minstd=1.0,
        maxstd=1.0,
        minlogstd=None,
        maxlogstd=None,
        validate_args=None,
    ):
        super().__init__()
        self.dist_type = dist_type
        self.minstd = minstd
        self.maxstd = maxstd
        self.minlogstd = minlogstd
        self.maxlogstd = maxlogstd
        self.validate_args = validate_args

    def forward(self, mu, logstd):
        if self.dist_type == 'normal':
            sigma = torch.exp(logstd)
            distr = D.Normal(mu, sigma, validate_args=self.validate_args)

        elif self.dist_type == 'squashed_normal':
            if self.minlogstd is not None or self.maxlogstd is not None:
                logstd = torch.clamp(logstd, self.minlogstd, self.maxlogstd)
            sigma = logstd.exp()
            distr = SquashedNormal(mu, sigma, validate_args=self.validate_args)

        elif self.dist_type == 'dreamerv3_normal':
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * torch.sigmoid(logstd + 2.0) + lo
            mu = torch.tanh(mu)
            distr = D.Normal(mu, std, validate_args=self.validate_args)
            sigma = std

        else:
            raise NotImplementedError(self.dist_type)

        return mu, sigma, distr

    def __repr__(self):
        return f'Dist(dist_type={self.dist_type})'
