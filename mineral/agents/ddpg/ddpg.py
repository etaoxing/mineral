import collections
import os
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...buffers import NStepReplay, ReplayBuffer
from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ..agent import Agent
from . import models
from .noise import add_mixed_normal_noise, add_normal_noise
from .schedule_util import ExponentialSchedule, LinearSchedule
from .utils import distl_projection, soft_update


class DDPG(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.ddpg_config = full_cfg.agent.ddpg
        self.num_actors = self.ddpg_config.num_actors
        self.max_agent_steps = int(self.ddpg_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        # --- Normalizers ---
        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Model ---
        obs_dim = self.obs_space['obs']
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim

        ActorCls = getattr(models, self.network_config.actor)
        CriticCls = getattr(models, self.network_config.critic)
        self.actor = ActorCls(obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.critic = CriticCls(obs_dim, self.action_dim, **self.network_config.get("critic_kwargs", {}))
        self.actor.to(self.device)
        self.critic.to(self.device)
        print('Actor:', self.actor)
        print('Critic:', self.critic, '\n')

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.ddpg_config.optim_type)
        self.actor_optim = OptimCls(
            self.actor.parameters(),
            **self.ddpg_config.get("actor_optim_kwargs", {}),
        )
        self.critic_optim = OptimCls(
            self.critic.parameters(),
            **self.ddpg_config.get("critic_optim_kwargs", {}),
        )
        print('Actor Optim:', self.actor_optim)
        print('Critic Optim:', self.critic_optim, '\n')

        # --- Target Networks ---
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.ddpg_config.no_tgt_actor else self.actor

        # --- Buffers ---
        self.memory = ReplayBuffer(
            self.obs_space, self.action_dim, capacity=int(self.ddpg_config.memory_size), device=self.device
        )
        self.n_step_buffer = NStepReplay(
            self.obs_space, self.action_dim, self.num_actors, self.ddpg_config.nstep, device=self.device
        )
        self.reward_shaper = RewardShaper(**self.ddpg_config.reward_shaper)

        # --- DDPG Exploration Noise Scheduler ---
        if self.ddpg_config.noise.decay == 'linear':
            self.noise_scheduler = LinearSchedule(
                start_val=self.ddpg_config.noise.std_max,
                end_val=self.ddpg_config.noise.std_min,
                total_iters=self.ddpg_config.noise.lin_decay_iters,
            )
        elif self.ddpg_config.noise.decay == 'exp':
            self.noise_scheduler = ExponentialSchedule(
                start_val=self.ddpg_config.noise.std_max,
                gamma=self.ddpg_config.exp_decay_rate,
                end_val=self.ddpg_config.noise.std_min,
            )
        else:
            self.noise_scheduler = None

    def get_actions(self, obs, sample=True):
        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        obs = obs['obs']
        mu, sigma, distr = self.actor(obs)
        if distr is None:
            actions = mu
        else:
            raise NotImplementedError
        if sample:
            if self.ddpg_config.noise.type == 'fixed':
                actions = add_normal_noise(actions, std=self.get_noise_std(), out_bounds=[-1.0, 1.0])
            elif self.ddpg_config.noise.type == 'mixed':
                actions = add_mixed_normal_noise(  # PQL mixed exploration
                    actions,
                    std_min=self.ddpg_config.noise.std_min,
                    std_max=self.ddpg_config.noise.std_max,
                    out_bounds=[-1.0, 1.0],
                )
            else:
                raise NotImplementedError
        return actions

    def get_noise_std(self):
        if self.noise_scheduler is None:
            return self.ddpg_config.noise.std_max
        else:
            return self.noise_scheduler.val()

    def update_noise(self):
        # TODO: currently unused
        if self.noise_scheduler is not None:
            self.noise_scheduler.step()

    @torch.no_grad()
    def get_tgt_policy_actions(self, obs, sample=True):
        mu, sigma, distr = self.actor_target(obs)
        if distr is None:
            actions = mu
        else:
            raise NotImplementedError
        if sample:
            # target policy smoothing (TD3)
            actions = add_normal_noise(
                actions,
                std=self.ddpg_config.noise.tgt_pol_std,
                noise_bounds=[-self.ddpg_config.noise.tgt_pol_noise_bound, self.ddpg_config.noise.tgt_pol_noise_bound],
                out_bounds=[-1.0, 1.0],
            )
        return actions

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, timesteps), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                for k, v in self.obs.items():
                    self.obs_rms[k].update(v)
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(self.obs, sample=sample)

            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if self.ddpg_config.handle_timeout:
                dones = self._handle_timeout(dones, infos)
            for k, v in self.obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_rewards[:, i] = rewards
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v
            self.obs = next_obs

        self.metrics.flush_video_buf(self.epoch)

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.num_actors

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

        self.set_eval()
        trajectory, steps = self.explore_env(self.env, self.ddpg_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.agent_steps += steps

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.ddpg_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            self.set_train()
            results = self.update_net(self.memory)

            # gather train metrics
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # episode metrics
            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_rewards.mean(),
                "train_scores/episode_lengths": self.metrics.episode_lengths.mean(),
            }
            metrics.update(episode_metrics)
            metrics = self.metrics.result(metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def update_net(self, memory):
        results = collections.defaultdict(list)
        for i in range(self.ddpg_config.mini_epochs):
            self.mini_epoch += 1
            obs, action, reward, next_obs, done = memory.sample_batch(self.ddpg_config.batch_size)

            if self.normalize_input:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
                next_obs = {k: self.obs_rms[k].normalize(v) for k, v in next_obs.items()}
            obs, next_obs = obs['obs'], next_obs['obs']

            critic_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, done)
            results["loss/critic"].append(critic_loss)
            results["grad_norm/critic"].append(critic_grad_norm)

            if self.mini_epoch % self.ddpg_config.update_actor_interval == 0:
                actor_loss, actor_grad_norm = self.update_actor(obs)
                results["loss/actor"].append(actor_loss)
                results["grad_norm/actor"].append(actor_grad_norm)

            if self.mini_epoch % self.ddpg_config.update_targets_interval == 0:
                soft_update(self.critic_target, self.critic, self.ddpg_config.tau)
                if not self.ddpg_config.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.ddpg_config.tau)
        return results

    def update_critic(self, obs, action, reward, next_obs, done):
        distl = getattr(self.critic, "distl", False)
        if distl:
            critic_loss_fn = F.binary_cross_entropy
        else:
            critic_loss_fn = F.mse_loss

        with torch.no_grad():
            next_actions = self.get_tgt_policy_actions(next_obs)
            if distl:
                target_Qs = self.critic_target.get_q_values(next_obs, next_actions)
                target_Qs_projected = [
                    distl_projection(
                        next_dist=target_Q,
                        reward=reward,
                        done=done,
                        gamma=self.ddpg_config.gamma**self.ddpg_config.nstep,
                        v_min=self.critic_target.v_min,
                        v_max=self.critic_target.v_max,
                        num_atoms=self.critic_target.num_atoms,
                        support=self.critic.z_atoms,
                    )
                    for target_Q in target_Qs
                ]
                target_Q = torch.min(torch.stack(target_Qs_projected), dim=0).values
            else:
                target_Q = self.critic_target.get_q_min(next_obs, next_actions)
                target_Q = reward + (1 - done) * (self.ddpg_config.gamma**self.ddpg_config.nstep) * target_Q

        current_Qs = self.critic.get_q_values(obs, action)
        critic_loss = torch.sum(torch.stack([critic_loss_fn(current_Q, target_Q) for current_Q in current_Qs]))
        grad_norm = self.optimizer_update(self.critic_optim, critic_loss)
        return critic_loss, grad_norm

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        mu, sigma, distr = self.actor(obs)
        if distr is None:
            action = mu
        else:
            raise NotImplementedError
        Q = self.critic.get_q_min(obs, action)
        actor_loss = -Q.mean()
        grad_norm = self.optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)
        return actor_loss, grad_norm

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.ddpg_config.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters=optimizer.param_groups[0]["params"],
                max_norm=self.ddpg_config.max_grad_norm,
            )
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm

    def eval(self):
        raise NotImplementedError

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
        if self.normalize_input:
            self.obs_rms.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        if self.normalize_input:
            self.obs_rms.eval()

    def save(self, f):
        pass

    def load(self, f):
        pass
