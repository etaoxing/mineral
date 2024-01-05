import collections
import itertools
import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...buffers import NStepReplay, ReplayBuffer
from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ..agent import Agent
from ..ddpg import models
from ..ddpg.utils import soft_update


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class SAC(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.sac_config = full_cfg.agent.sac
        self.num_actors = self.sac_config.num_actors
        self.max_agent_steps = int(self.sac_config.max_agent_steps)
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

        # --- Encoder ---
        if self.network_config.get("encoder", None) is not None:
            EncoderCls = getattr(models, self.network_config.encoder)
            self.encoder = EncoderCls(**self.network_config.get("encoder_kwargs", {}))
        else:
            self.encoder = Lambda(lambda x: x["obs"])
        self.encoder.to(self.device)
        print('Encoder:', self.encoder)

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
        OptimCls = getattr(torch.optim, self.sac_config.optim_type)
        self.actor_optim = OptimCls(
            itertools.chain(self.encoder.parameters(), self.actor.parameters()),
            **self.sac_config.get("actor_optim_kwargs", {}),
        )
        self.critic_optim = OptimCls(
            itertools.chain(self.encoder.parameters(), self.critic.parameters()),
            **self.sac_config.get("critic_optim_kwargs", {}),
        )
        print('Actor Optim:', self.actor_optim)
        print('Critic Optim:', self.critic_optim, '\n')

        # --- Target Networks ---
        self.encoder_target = deepcopy(self.encoder)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.sac_config.no_tgt_actor else self.actor

        # --- Buffers ---
        self.memory = ReplayBuffer(
            self.obs_space, self.action_dim, capacity=int(self.sac_config.memory_size), device=self.device
        )
        self.n_step_buffer = NStepReplay(
            self.obs_space, self.action_dim, self.num_actors, self.sac_config.nstep, device=self.device
        )
        self.reward_shaper = RewardShaper(**self.sac_config.reward_shaper)

        # --- SAC Entropy ---
        self.target_entropy = -self.action_dim
        if self.sac_config.alpha is None:
            init_alpha = np.log(self.sac_config.init_alpha)
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha, device=self.device, dtype=torch.float32))
            self.alpha_optim = OptimCls([self.log_alpha], **self.sac_config.get("alpha_optim_kwargs", {}))

    def get_actions(self, obs=None, z=None, sample=True, logprob=False):
        if z is None:
            assert obs is not None
            if self.normalize_input:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
            z = self.encoder(obs)

        mu, sigma, distr = self.actor(z)
        assert distr is not None

        if sample:
            actions = distr.rsample()
        else:
            actions = mu

        if logprob:
            log_prob = distr.log_prob(actions).sum(-1, keepdim=True)
            return actions, distr, log_prob
        else:
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
                actions = self.get_actions(obs=self.obs, sample=sample)

            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if self.sac_config.handle_timeout:
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
        trajectory, steps = self.explore_env(self.env, self.sac_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.agent_steps += steps

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.sac_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            self.set_train()
            results = self.update_net(self.memory)

            # gather train metrics
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch, "alpha": self.get_alpha(scalar=True)})
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
        for i in range(self.sac_config.mini_epochs):
            self.mini_epoch += 1
            obs, action, reward, next_obs, done = memory.sample_batch(self.sac_config.batch_size)

            critic_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, done)
            results["loss/critic"].append(critic_loss)
            results["grad_norm/critic"].append(critic_grad_norm)

            if self.mini_epoch % self.sac_config.update_actor_interval == 0:
                actor_loss, alpha_loss, actor_grad_norm = self.update_actor(obs)
                results["loss/actor"].append(actor_loss)
                results["loss/alpha"].append(alpha_loss)
                results["grad_norm/actor"].append(actor_grad_norm)

            if self.mini_epoch % self.sac_config.update_targets_interval == 0:
                soft_update(self.critic_target, self.critic, self.sac_config.tau)
                if not self.sac_config.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.sac_config.tau)
        return results

    def get_alpha(self, detach=True, scalar=False):
        if self.sac_config.alpha is None:
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
        else:
            alpha = self.sac_config.alpha
        return alpha

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            if self.normalize_input:
                next_obs = {k: self.obs_rms[k].normalize(v) for k, v in next_obs.items()}
            next_z = self.encoder_target(next_obs)
            next_actions, _, log_prob = self.get_actions(z=next_z, logprob=True)
            target_Q = self.critic_target.get_q_min(next_z, next_actions) - self.get_alpha() * log_prob
            target_Q = reward + (1 - done) * (self.sac_config.gamma**self.sac_config.nstep) * target_Q

        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        z = self.encoder(obs)
        current_Qs = self.critic.get_q_values(z, action)
        critic_loss = torch.sum(torch.stack([F.mse_loss(current_Q, target_Q) for current_Q in current_Qs]))
        grad_norm = self.optimizer_update(self.critic_optim, critic_loss)
        return critic_loss, grad_norm

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        z = self.encoder(obs)
        z = z.detach()  # sac_ae
        actions, _, log_prob = self.get_actions(z=z, logprob=True)
        Q = self.critic.get_q_min(z, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()
        grad_norm = self.optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)

        if self.sac_config.alpha is None:
            alpha_loss = (self.get_alpha(detach=False) * (-log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss, alpha_loss, grad_norm

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.sac_config.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters=optimizer.param_groups[0]["params"],
                max_norm=self.sac_config.max_grad_norm,
            )
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm

    def eval(self):
        raise NotImplementedError

    def set_train(self):
        if self.normalize_input:
            self.obs_rms.train()
        self.encoder.train()
        self.actor.train()
        self.critic.train()
        self.encoder_target.train()
        self.actor_target.train()
        self.critic_target.train()

    def set_eval(self):
        if self.normalize_input:
            self.obs_rms.eval()
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.encoder_target.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def save(self, f):
        pass

    def load(self, f):
        pass
