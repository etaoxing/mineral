import functools
import os
import pprint
import sys

import hydra
import numpy as np
import wandb
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def make_envs(config):
    return make_isaacgym_envs(config)


def make_datasets(config, env):
    return None


def import_isaacgym():
    # https://github.com/NVlabs/sim-web-visualizer/blob/main/example/isaacgym/train_isaacgym_remote_server.ipynb
    import os
    import sys
    from ctypes import cdll
    from pathlib import Path

    is_conda = 'CONDA_PREFIX' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ
    if is_conda:
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor >= 8:
            conda_lib_path = (
                Path(sys.executable).parent.parent / f"lib/libpython{version_info.major}.{version_info.minor}.so.1.0"
            )
        else:
            conda_lib_path = (
                Path(sys.executable).parent.parent / f"lib/libpython{version_info.major}.{version_info.minor}m.so.1.0"
            )
        python_lib = cdll.LoadLibrary(str(conda_lib_path))
        print(f"Load Python lib {conda_lib_path}")

    import isaacgym
    import torch


def make_isaacgym_envs(config):
    # ---

    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict

    # `xvfb-run -a -s "-screen 0 256x256x24 -ac +extension GLX +render -noreset" python ...`
    # set virtual_screen_capture=True and headless=False to get IsaacGym GUI window
    # if you get `OSError: Pillow was built without XCB support`, then `pip install -U Pillow`
    # (switch to pip instead of conda package)  # https://stackoverflow.com/a/66682282
    # TODO: https://github.com/NVlabs/sim-web-visualizer/tree/main/example/isaacgym

    if config.env_render:
        headless, virtual_screen_capture = False, True
    else:
        headless, virtual_screen_capture = config.headless, False

    env = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=headless,
        virtual_screen_capture=virtual_screen_capture,
        force_render=False,
    )
    return env


def save_run_metadata(logdir, run_name, run_id, resolved_config):
    run_metadata = {
        'logdir': logdir,
        'run_name': run_name,
        'run_id': run_id,
    }
    yaml.dump(run_metadata, open(os.path.join(logdir, 'run_metadata.yaml'), 'w'), default_flow_style=False)
    yaml.dump(resolved_config, open(os.path.join(logdir, 'resolved_config.yaml'), 'w'), default_flow_style=False)


def main(config: DictConfig):
    if 'isaacgym' in sys.modules or 'isaacgymenvs' in sys.modules:
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    else:
        from .utils import set_np_formatting, set_seed

    from .utils import limit_threads

    limit_threads(1)

    # set numpy formatting for printing only
    set_np_formatting()

    # ---------- Setup Run ----------
    logdir = config.logdir
    os.makedirs(logdir, exist_ok=True)

    if config.ckpt:
        config.ckpt = to_absolute_path(config.ckpt)

    if config.multi_gpu:
        from accelerate import Accelerator

        accelerator = Accelerator()
        rank = int(os.getenv('LOCAL_RANK', '0'))
        if str(accelerator.device) == 'cuda':
            pass
        else:
            assert rank == accelerator.device.index, print(rank, accelerator.device, accelerator.device.index)

        config.sim_device = f'cuda:{rank}'
        config.rl_device = f'cuda:{rank}'
        config.graphics_device_id = rank
        # sets seed. if seed is -1 will pick a random one
        config.seed = set_seed(config.seed + rank)

        # if rank != 0:
        #     f = open(os.path.join(config.logdir, 'log_{rank}.txt', 'w'))
        #     sys.stdout = f
    else:
        accelerator = None
        rank = 0

        # use the same device for sim and rl
        config.sim_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.graphics_device_id = config.device_id if config.device_id >= 0 else 0
        config.seed = set_seed(config.seed)

    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print(pprint.pformat(resolved_config, compact=True, indent=1), '\n')

    if rank == 0:
        os.environ['WANDB_START_METHOD'] = 'thread'
        # connect to wandb
        wandb_config = OmegaConf.to_container(config.wandb, resolve=True)
        wandb_run = wandb.init(
            **wandb_config,
            dir=logdir,
            config=resolved_config,
        )
        run_name, run_id = wandb_run.name, wandb_run.id
        print(f'run_name: {run_name}, run_id: {run_id}')
        save_run_metadata(logdir, run_name, run_id, resolved_config)

    # ---------- Run Agent ----------
    cprint('Making Envs', 'green', attrs=['bold'])
    env = make_envs(config)
    print('-' * 20)
    print(f'Env: {env}')

    datasets = make_datasets(config, env)
    print(f'Datasets: {datasets}')

    from .. import agents

    AgentCls = getattr(agents, config.agent.algo)
    print(f'AgentCls: {AgentCls}')
    agent = AgentCls(env, logdir, config, accelerator=accelerator, datasets=datasets)

    if config.ckpt:
        print(f'Loading checkpoint: {config.ckpt}')
        agent.load(config.ckpt, ckpt_keys=config.ckpt_keys)

    if config.run == 'train':
        agent.train()
    elif config.run == 'eval':
        agent.eval()
    elif config.run == 'train_eval':
        agent.train()
        agent.load(os.path.join(agent.ckpt_dir, 'final.pth'))
        agent.eval()
    else:
        raise NotImplementedError(config.run)

    if rank == 0:
        # close wandb
        wandb.finish()


if __name__ == '__main__':
    c = []
    hydra.main(
        config_name='config',
        config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../cfgs'),
        version_base='1.1',
    )(lambda x: c.append(x))()
    config = c[0]

    import_isaacgym()  # uncomment for isaacgym (need to import before torch)

    main(config)
