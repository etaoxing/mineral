import math

import numpy as np
import torch


class RewardShaper:
    def __init__(
        self,
        fn='scale',
        scale=1.0,
    ):
        self.fn = fn
        self.scale = scale

    def __call__(self, rewards):
        if self.fn == 'scale':
            rewards *= self.scale
        else:
            raise NotImplementedError(self.fn)
        return rewards


class AdaptiveScheduler:
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)


def adjust_learning_rate_cos(init_lr, epoch, mini_epochs, agent_steps, max_agent_steps):
    lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * (agent_steps + epoch / mini_epochs) / max_agent_steps))
    return lr
