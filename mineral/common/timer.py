import collections
import contextlib
import time

import numpy as np


class Timer:
    def __init__(self, columns=('count', 'sum', 'frac', 'avg', 'min', 'max', 'total', 'totalrate', 'lastrate')):
        available = ('count', 'sum', 'frac', 'avg', 'min', 'max', 'total', 'totalrate', 'lastrate')
        assert all(x in available for x in columns), columns
        self.columns = columns
        self._active_scopes = {}
        self._durations = collections.defaultdict(list)
        self._totals = collections.defaultdict(float)
        self._last = time.perf_counter()
        self._step = 0

    def reset(self):
        for k, v in self._active_scopes.items():
            v.__exit__(None, None, None)
        self._active_scopes.clear()
        for timings in self._durations.values():
            timings.clear()
        self._last = time.perf_counter()

    @contextlib.contextmanager
    def scope(self, name):
        start = time.perf_counter()
        yield
        stop = time.perf_counter()
        dur = stop - start
        self._durations[name].append(dur)

    def wrap(self, name, obj, methods):
        for method in methods:
            decorator = self.scope(f'{name}.{method}')
            setattr(obj, method, decorator(getattr(obj, method)))

    def start(self, name):
        if name in self._active_scopes:
            raise ValueError(f"Scope '{name}' is already active")
        self._active_scopes[name] = self.scope(name)
        self._active_scopes[name].__enter__()

    def end(self, name):
        if name not in self._active_scopes:
            raise ValueError(f"Scope '{name}' is not active.")
        self._active_scopes[name].__exit__(None, None, None)
        del self._active_scopes[name]

    def stats(self, step=None, reset=True, total_names=None):
        now = time.perf_counter()
        passed = now - self._last
        metrics = {}
        metrics['duration'] = passed
        for name, durs in self._durations.items():
            available = {}
            available['count'] = len(durs)
            available['sum'] = np.sum(durs)
            available['frac'] = np.sum(durs) / metrics['duration']
            if len(durs):
                available['avg'] = np.mean(durs)
                available['min'] = np.min(durs)
                available['max'] = np.max(durs)
            self._totals[name] += available['sum']
            available['total'] = self._totals[name]
            if step is not None:
                available['totalrate'] = step / available['total']
                available['lastrate'] = (step - self._step) / available['sum']
            for key, value in available.items():
                if key in self.columns:
                    metrics[f'{name}/{key}'] = value
        if total_names is not None:
            metrics['total'] = sum([self._totals[k] for k in total_names])  # assuming no overlap
            if step is not None:
                metrics['totalrate'] = step / metrics['total']
                metrics['lastrate'] = (step - self._step) / metrics['duration']
        if reset:
            if step is not None:
                self._step = step
            self.reset()
        else:
            self._totals.clear()
        return metrics
