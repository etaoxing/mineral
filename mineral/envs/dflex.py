import os
import sys

from omegaconf import OmegaConf

DEFAULT_DFLEXENVS_KWARGS = {
    'ant': dict(env_name='AntEnv', episode_length=1000, MM_caching_frequency=16),
    'cartpole': dict(env_name='CartPoleSwingUpEnv', episode_length=240, MM_caching_frequency=4),
    'cheetah': dict(env_name='CheetahEnv', episode_length=1000, MM_caching_frequency=16),
    'hopper': dict(env_name='HopperEnv', episode_length=1000, MM_caching_frequency=16),
    'humanoid': dict(env_name='HumanoidEnv', episode_length=1000, MM_caching_frequency=48),
    'snu_humanoid': dict(env_name='SNUHumanoidEnv', episode_length=1000, MM_caching_frequency=8),
}


def make_envs(config):
    env = config.task.env
    env_kwargs = OmegaConf.to_container(env, resolve=True)
    env_name, num_envs = env_kwargs.pop('env_name'), env_kwargs.pop('numEnvs')

    env_kwargs = {
        **DEFAULT_DFLEXENVS_KWARGS[env_name],
        **env_kwargs,
    }
    env_name = env_kwargs.pop('env_name')

    DIFFRL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/DiffRL'))
    sys.path.append(DIFFRL_PATH)

    import envs as DFlexEnvs

    env_fn = getattr(DFlexEnvs, env_name)
    env = env_fn(
        num_envs=num_envs,
        device=config.sim_device,
        seed=config.seed,
        render=config.env_render,
        **env_kwargs,
    )

    return env
