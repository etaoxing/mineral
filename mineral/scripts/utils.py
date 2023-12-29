import os
import sys
from typing import Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
except:
    pass  # ignore if already registered


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def set_np_formatting():
    """formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr='inf',
        linewidth=4000,
        nanstr='nan',
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def set_seed(seed, torch_deterministic=False, rank=0):
    import random

    import numpy as np
    import torch

    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # BUG: https://discuss.pytorch.org/t/deterministic-algorithms-yield-an-error/181809
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def limit_threads(n: int = 1):
    # blosc.set_nthreads(n)
    # if n == 1:
    #     blosc.use_threads = False
    # torch.set_num_threads(n)
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)


def make_batch_env(num_envs, make_env_fn, parallel='process', device='numpy'):
    from functools import partial as bind

    from ..common.batch_env import BatchEnv
    from ..common.parallel import Parallel

    ctors = []
    for index in range(num_envs):
        ctor = lambda: make_env_fn()
        if parallel != 'none':
            ctor = bind(Parallel, ctor, parallel)
        ctors.append(ctor)

    envs = [ctor() for ctor in ctors]
    env = BatchEnv(envs, parallel=parallel != 'none', device=device)
    return env
