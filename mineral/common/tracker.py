from collections import deque
from collections.abc import Sequence

import numpy as np
import torch


class Tracker:
    def __init__(self, max_len):
        self.window = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.window.__repr__()

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.window.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.window.extend(value)
        else:
            self.window.append(value)

    def mean(self):
        return np.mean(self.window)

    def std(self):
        return np.std(self.window)

    def max(self):
        return np.max(self.window)

    def min(self):
        return np.min(self.window)
