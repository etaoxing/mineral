import collections
import itertools
import json
import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ... import nets
from ...buffers import NStepReplay, ReplayBuffer
from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ..agent import Agent
from ..ddpg import models
from ..ddpg.utils import soft_update, weight_init_


class SAC(Agent):
    r"""Soft Actor-Critic."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.sac_config = full_cfg.agent.sac
        self.num_actors = self.sac_config.num_actors
        self.max_agent_steps = int(self.sac_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        # --- Normalizers ---
        rms_config = dict(eps=1e-4, with_clamp=True, initial_count="eps")
        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Encoder ---
        if self.network_config.get("encoder", None) is not None:
            EncoderCls = getattr(nets, self.network_config.encoder)
            encoder_kwargs = self.network_config.get("encoder_kwargs", {})
            self.encoder = EncoderCls(self.obs_space, encoder_kwargs, weight_init_fn=weight_init_)
        else:
            f = lambda x: x['obs']
            self.encoder = nets.Lambda(f)
        self.encoder.to(self.device)
        print('Encoder:', self.encoder)

        # --- Model ---
        if self.network_config.get("encoder", None) is not None:
            obs_dim = self.encoder.out_dim
        else:
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
        self.encoder_target = self.encoder
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
        target_entropy_scalar = self.sac_config.get("target_entropy_scalar", 1.0)
        self.target_entropy = -self.action_dim * target_entropy_scalar
        # RLPD divides by 2, https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py#L78-L79
        print('Target Entropy Scalar:', target_entropy_scalar, 'Target Entropy:', self.target_entropy)
        if self.sac_config.alpha is None:
            init_alpha = np.log(self.sac_config.init_alpha)
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha, device=self.device, dtype=torch.float32))
            self.alpha_optim = OptimCls([self.log_alpha], **self.sac_config.get("alpha_optim_kwargs", {}))

        # --- Timing ---
        self.timer = Timer()
        self.timer.wrap("agent", self, ["explore_env", "update_net"])
        self.timer.wrap("memory", self.memory, ["add_to_buffer"])
        self.timer.wrap("env", self.env, ["step"])
        self.timer_total_names = ("agent.explore_env", "memory.add_to_buffer", "agent.update_net")

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
            self.obs = next_obs  # update obs

        self.metrics.flush_video(self.epoch)

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

            # train metrics
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch, "alpha": self.get_alpha(scalar=True)})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # timing metrics
            timings = self.timer.stats(step=self.agent_steps, total_names=self.timer_total_names, reset=False)
            timing_metrics = {f'train_timings/{k}': v for k, v in timings.items()}
            metrics.update(timing_metrics)

            # episode metrics
            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
            }
            metrics.update(episode_metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(metrics["train_scores/episode_rewards"])

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f'Epochs: {self.epoch + 1} |',
                    f'Agent Steps: {int(self.agent_steps):,} |',
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    f'Stats:',
                    f'ep_rewards {episode_metrics["train_scores/episode_rewards"]:.2f},',
                    f'ep_lengths {episode_metrics["train_scores/episode_lengths"]:.2f},',
                    f'last_sps {timings["lastrate"]:.2f},',
                    f'SPS {timings["totalrate"]:.2f} |',
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def update_net(self, memory):
        results = collections.defaultdict(list)
        for i in range(self.sac_config.mini_epochs):
            self.mini_epoch += 1
            obs, action, reward, next_obs, done = memory.sample_batch(self.sac_config.batch_size, device=self.device)

            critic_loss, critic_grad_norm, target_values = self.update_critic(obs, action, reward, next_obs, done)
            results["loss/critic"].append(critic_loss)
            results["grad_norm/critic"].append(critic_grad_norm)
            for k, v in target_values.items():
                results[k].append(v)

            if self.mini_epoch % self.sac_config.update_actor_interval == 0:
                actor_loss, alpha_loss, entropy, actor_grad_norm = self.update_actor(obs)
                results["loss/actor"].append(actor_loss)
                results["loss/alpha"].append(alpha_loss)
                results["entropy"].append(entropy)
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
            target_Q = self.critic_target.get_q_min(next_z, next_actions)
            if self.sac_config.backup_entropy:
                # https://github.com/ikostrikov/jaxrl/blob/4a9abaff1c915b00ca73d35a30faf16b165e52ec/jaxrl/agents/sac/critic.py#L29
                target_Q -= self.get_alpha() * log_prob
            target_Q = reward + (1 - done) * (self.sac_config.gamma**self.sac_config.nstep) * target_Q

        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        z = self.encoder(obs)
        current_Qs = self.critic.get_q_values(z, action)
        critic_loss = torch.sum(torch.stack([F.mse_loss(current_Q, target_Q) for current_Q in current_Qs]))
        grad_norm = self.optimizer_update(self.critic_optim, critic_loss)

        target_values = {
            "target_values/mean": target_Q.mean(),
            "target_values/std": target_Q.std(),
            "target_values/max": target_Q.max(),
            "target_values/min": target_Q.min(),
        }

        return critic_loss, grad_norm, target_values

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        z = self.encoder(obs)
        if self.sac_config.get("actor_detach_encoder", False):
            z = {k: v.detach() for k, v in z.items()} if isinstance(z, dict) else z.detach()  # sac_ae
        actions, _, log_prob = self.get_actions(z=z, logprob=True)
        Q = self.critic.get_q_min(z, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()
        grad_norm = self.optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)

        entropy = -log_prob
        if self.sac_config.alpha is None:
            alpha_loss = (self.get_alpha(detach=False) * (entropy - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss, alpha_loss, entropy.mean(), grad_norm

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
        self.set_eval()

        obs = self.env.reset()
        obs = self._convert_obs(obs)
        dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        sample = True
        total_eval_episodes = self.num_actors * 2
        eval_metrics = self._create_metrics(total_eval_episodes, self.metrics_kwargs)
        with self._as_metrics(eval_metrics), torch.no_grad():
            while self.metrics.num_episodes < total_eval_episodes:
                if not self.env_autoresets:
                    raise NotImplementedError

                actions = self.get_actions(obs=obs, sample=sample)
                next_obs, rewards, dones, infos = self.env.step(actions)
                next_obs = self._convert_obs(next_obs)
                rewards, dones = (
                    torch.as_tensor(rewards, device=self.device),
                    torch.as_tensor(dones, device=self.device),
                )

                done_indices = torch.where(dones)[0].tolist()
                self.metrics.update(self.epoch, self.env, obs, rewards, done_indices, infos)

                obs = next_obs  # update obs
            self.metrics.flush_video(self.epoch)

            metrics = {
                "eval_scores/num_episodes": self.metrics.num_episodes,
                "eval_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "eval_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                **self.metrics.result(prefix="eval"),
            }
            print(metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            scores = {
                "epoch": self.epoch,
                "mini_epoch": self.mini_epoch,
                "agent_steps": self.agent_steps,
                "eval_scores/num_episodes": self.metrics.num_episodes,
                "eval_scores/episode_rewards": list(self.metrics.episode_trackers["rewards"].window),
                "eval_scores/episode_lengths": list(self.metrics.episode_trackers["lengths"].window),
            }
            json.dump(scores, open(os.path.join(self.logdir, "scores.json"), "w"), indent=4)

    def set_train(self):
        self.encoder.train()
        self.actor.train()
        self.critic.train()
        self.encoder_target.train()
        self.actor_target.train()
        self.critic_target.train()

    def set_eval(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.encoder_target.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def save(self, f):
        ckpt = {
            'epoch': self.epoch,
            'mini_epoch': self.mini_epoch,
            'agent_steps': self.agent_steps,
            'obs_rms': self.obs_rms.state_dict() if self.normalize_input else None,
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'encoder_target': self.encoder_target.state_dict(),
            'actor_target': self.actor_target.state_dict() if not self.sac_config.no_tgt_actor else None,
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.data if self.sac_config.alpha is None else None,
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        all_ckpt_keys = ('epoch', 'mini_epoch', 'agent_steps')
        all_ckpt_keys += ('obs_rms', 'encoder', 'actor', 'critic')
        all_ckpt_keys += ('encoder_target', 'actor_target', 'critic_target')
        all_ckpt_keys += ('log_alpha',)
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f'Warning: ckpt skipped loading `{k}`')
                continue
            if k == 'obs_rms' and (not self.normalize_input):
                continue
            if k == 'actor_target' and (self.sac_config.no_tgt_actor):
                continue
            if k == 'log_alpha' and (not self.sac_config.alpha is None):
                continue

            if k == 'log_alpha':
                self.log_alpha.data = ckpt[k]
                continue

            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
