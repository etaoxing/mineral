# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import collections
import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ...common.tracker import Tracker
from ..actorcritic_base import ActorCriticBase
from . import models
from .utils import CriticDataset, grad_norm


class SHAC(ActorCriticBase):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.shac_config = full_cfg.agent.shac
        self.num_actors = self.shac_config.num_actors
        self.max_agent_steps = int(self.shac_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length

        # --- SHAC Parameters ---
        self.gamma = self.shac_config.get('gamma', 0.99)
        self.critic_method = self.shac_config.get('critic_method', 'one-step')  # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = self.shac_config.get('lambda', 0.95)
        self.critic_iterations = self.shac_config.get('critic_iterations', 16)
        self.target_critic_alpha = self.shac_config.get('target_critic_alpha', 0.4)

        self.horizon_len = self.shac_config.horizon_len
        self.max_epochs = self.shac_config.max_epochs
        self.num_batch = self.shac_config.get('num_batch', 4)
        self.batch_size = self.num_envs * self.horizon_len // self.num_batch

        # --- Normalizers ---
        rms_config = dict(eps=1e-5, correction=0, initial_count=1e-4, dtype=torch.float64)  # unbiased=False -> correction=0
        if self.shac_config.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        self.ret_rms = None
        if self.shac_config.normalize_ret:
            self.ret_rms = normalizers.RunningMeanStd((), **rms_config).to(self.device)

        # --- Model ---
        obs_dim = self.obs_space['obs']
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
        assert obs_dim == self.env.num_obs
        assert self.action_dim == self.env.num_actions

        ActorCls = getattr(models, self.network_config.actor)
        CriticCls = getattr(models, self.network_config.critic)
        self.actor = ActorCls(obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.critic = CriticCls(obs_dim, self.action_dim, **self.network_config.get("critic_kwargs", {}))
        self.actor.to(self.device)
        self.critic.to(self.device)
        print('Actor:', self.actor)
        print('Critic:', self.critic, '\n')

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.shac_config.optim_type)
        self.actor_optim = OptimCls(
            self.actor.parameters(),
            **self.shac_config.get("actor_optim_kwargs", {}),
        )
        self.critic_optim = OptimCls(
            self.critic.parameters(),
            **self.shac_config.get("critic_optim_kwargs", {}),
        )
        print('Actor Optim:', self.actor_optim)
        print('Critic Optim:', self.critic_optim, '\n')

        self.actor_lr = self.actor_optim.defaults["lr"]
        self.critic_lr = self.critic_optim.defaults["lr"]

        # --- Target Networks ---
        self.critic_target = deepcopy(self.critic)

        # --- Replay Buffer ---
        assert self.num_actors == self.env.num_envs
        T, B = self.horizon_len, self.num_envs
        self.create_buffers(T, B)

        self.reward_shaper = RewardShaper(**self.shac_config.reward_shaper)

        # --- Episode Metrics ---
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=int)
        self.episode_discounted_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        self.episode_rewards_hist = []
        self.episode_lengths_hist = []
        self.episode_discounted_rewards_hist = []

        tracker_len = 100
        self.episode_rewards_tracker = Tracker(tracker_len)
        self.episode_lengths_tracker = Tracker(tracker_len)
        self.episode_discounted_rewards_tracker = Tracker(tracker_len)

        # --- Timing ---
        self.timer = Timer()

    def create_buffers(self, T, B):
        self.obs_buf = {k: torch.zeros((T, B) + v, dtype=torch.float32, device=self.device) for k, v in self.obs_space.items()}
        self.rew_buf = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.ret = torch.zeros((B), dtype=torch.float32, device=self.device)

        # for kl divergence computing
        self.old_mus = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)
        self.old_sigmas = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)
        self.mus = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)
        self.sigmas = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)

    def get_actions(self, obs, sample=True):
        # NOTE: obs_rms.normalize(...) occurs elsewhere
        obs = obs['obs']
        mu, sigma, distr = self.actor(obs)
        if sample:
            actions = distr.rsample()
        else:
            actions = mu
        # clamp actions
        actions = torch.tanh(actions)
        return actions

    @torch.no_grad()
    def evaluate_policy(self, num_episodes, deterministic=False):
        episode_rewards_hist = []
        episode_lengths_hist = []
        episode_discounted_rewards_hist = []
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, dtype=int)
        episode_discounted_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        obs = self.env.reset()
        obs = self._convert_obs(obs)

        episodes = 0
        while episodes < num_episodes:
            if self.obs_rms is not None:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

            actions = self.get_actions(obs, sample=not deterministic)
            obs, rew, done, _ = self.env.step(actions)
            obs = self._convert_obs(obs)

            episode_rewards += rew
            episode_lengths += 1
            episode_discounted_rewards += episode_gamma * rew
            episode_gamma *= self.gamma

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('rew = {:.2f}, len = {}'.format(episode_rewards[done_env_id].item(), episode_lengths[done_env_id]))
                    episode_rewards_hist.append(episode_rewards[done_env_id].item())
                    episode_lengths_hist.append(episode_lengths[done_env_id].item())
                    episode_discounted_rewards_hist.append(episode_discounted_rewards[done_env_id].item())
                    episode_rewards[done_env_id] = 0.0
                    episode_lengths[done_env_id] = 0
                    episode_discounted_rewards[done_env_id] = 0.0
                    episode_gamma[done_env_id] = 1.0
                    episodes += 1

        mean_episode_rewards = np.mean(np.array(episode_rewards_hist))
        mean_episode_lengths = np.mean(np.array(episode_lengths_hist))
        mean_episode_discounted_rewards = np.mean(np.array(episode_discounted_rewards_hist))

        return mean_episode_rewards, mean_episode_lengths, mean_episode_discounted_rewards

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    def train(self):
        # initializations
        self.initialize_env()

        while self.epoch < self.max_epochs:
            self.epoch += 1

            # learning rate schedule
            if self.shac_config.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(self.epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(self.epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optim.param_groups:
                    param_group['lr'] = critic_lr
            elif self.shac_config.lr_schedule == 'constant':
                lr = self.actor_lr
            else:
                raise NotImplementedError(self.shac_config.lr_schedule)

            # train actor
            self.timer.start("train/update_actor")
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            # train critic
            # prepare dataset
            self.timer.start("train/make_critic_dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last=False)
            self.timer.end("train/make_critic_dataset")

            self.timer.start("train/update_critic")
            critic_results = self.update_critic(dataset)
            self.timer.end("train/update_critic")

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1.0 - alpha) * param.data)

            # gather metrics
            results = {**actor_results, **critic_results}
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics = {"epoch": self.epoch, "lr": lr, **metrics}
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            timings_total_names = ("train/update_actor", "train/make_critic_dataset", "train/update_critic")
            timings = self.timer.stats(step=self.agent_steps, total_names=timings_total_names, reset=False)
            metrics = {**metrics, **{f"train_timings/{k}": v for k, v in timings.items()}}

            if len(self.episode_rewards_hist) > 0:
                mean_episode_rewards = self.episode_rewards_tracker.mean()
                mean_episode_lengths = self.episode_lengths_tracker.mean()
                mean_episode_discounted_rewards = self.episode_discounted_rewards_tracker.mean()

                episode_metrics = {
                    "train_metrics/episode_rewards": mean_episode_rewards,
                    "train_metrics/episode_lengths": mean_episode_lengths,
                    "train_metrics/episode_discounted_rewards": mean_episode_discounted_rewards,
                }
                metrics.update(episode_metrics)
            else:
                mean_episode_rewards = -np.inf
                mean_episode_lengths = 0
                mean_episode_discounted_rewards = -np.inf

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(mean_episode_rewards)

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f'Epoch: {self.epoch} |',
                    f'Agent Steps: {int(self.agent_steps):,} |',
                    f'SPS: {timings["lastrate"]:.2f} |',
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    f'Stats:',
                    f'ep_rewards {mean_episode_rewards:.2f},',
                    f'ep_lenths {mean_episode_lengths:.2f},',
                    f'ep_discounted_rewards {mean_episode_discounted_rewards:.2f},',
                    f'value_loss {metrics["train_stats/value_loss"]:.4f},',
                    f'grad_norm_before_clip {metrics["train_stats/grad_norm_before_clip"]:.2f},',
                    f'grad_norm_after_clip {metrics["train_stats/grad_norm_after_clip"]:.2f},',
                    f'\b\b |',
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

        # save reward/length history
        self.episode_rewards_hist = np.array(self.episode_rewards_hist)
        self.episode_lengths_hist = np.array(self.episode_lengths_hist)
        self.episode_discounted_rewards_hist = np.array(self.episode_discounted_rewards_hist)
        np.save(open(os.path.join(self.logdir, 'ep_rewards_hist.npy'), 'wb'), self.episode_rewards_hist)
        np.save(open(os.path.join(self.logdir, 'ep_lengths_hist.npy'), 'wb'), self.episode_lengths_hist)
        np.save(open(os.path.join(self.logdir, 'ep_discounted_rewards_hist.npy'), 'wb'), self.episode_discounted_rewards_hist)

    def update_actor(self):
        results = collections.defaultdict(list)

        def actor_closure():
            self.actor_optim.zero_grad()
            self.timer.start("train/actor_closure/actor_loss")

            self.timer.start("train/actor_closure/forward_sim")
            actor_loss = self.compute_actor_loss()
            self.timer.end("train/actor_closure/forward_sim")

            self.timer.start("train/actor_closure/backward_sim")
            actor_loss.backward()
            self.timer.end("train/actor_closure/backward_sim")

            with torch.no_grad():
                grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.shac_config.truncate_grads:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.shac_config.max_grad_norm)
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1000000.0:
                    print('NaN gradient')
                    raise ValueError

            results["actor_loss"].append(actor_loss)
            results["grad_norm_before_clip"].append(grad_norm_before_clip)
            results["grad_norm_after_clip"].append(grad_norm_after_clip)
            self.timer.end("train/actor_closure/actor_loss")
            return actor_loss

        self.actor_optim.step(actor_closure)
        return results

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                # TODO: not using mean centering of ret_rms?
                ret_var = self.ret_rms.running_var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            # normalize the current obs
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        # collect trajectories and compute actor loss
        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for i in range(self.horizon_len):
            # collect data for critic training
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            # take env step
            actions = self.get_actions(obs, sample=not deterministic)
            obs, rew, done, extra_info = self.env.step(actions)
            obs = self._convert_obs(obs)

            with torch.no_grad():
                raw_rew = rew.clone()
            # scale the reward
            rew = self.reward_shaper(rew)

            # update episode metrics
            with torch.no_grad():
                self.episode_rewards += raw_rew
                self.episode_lengths += 1
                self.episode_discounted_rewards += self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma

            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    for k, v in obs.items():
                        self.obs_rms[k].update(v)
                # normalize the current obs
                obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                # normalize the current rew
                rew = rew / torch.sqrt(ret_var + 1e-6)

            # value bootstrap when episode terminates
            next_values[i + 1] = self.critic_target(obs['obs']).squeeze(-1)
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                terminal_obs = extra_info['obs_before_reset']
                terminal_obs = self._convert_obs(terminal_obs)

                for id in done_env_ids:
                    nan = False
                    # TODO: some elements of obs_dict (for logging) may be nan, add regex to ignore these
                    for k, v in terminal_obs.items():
                        if (
                            (torch.isnan(v[id]).sum() > 0)
                            or (torch.isinf(v[id]).sum() > 0)
                            or ((torch.abs(v[id]) > 1e6).sum() > 0)
                        ):  # ugly fix for nan values
                            print(f'nan value: {k}')
                            nan = True
                            break

                    if nan:
                        next_values[i + 1, id] = 0.0
                    elif self.episode_lengths[id] < self.max_episode_length:  # early termination
                        next_values[i + 1, id] = 0.0
                    else:  # otherwise, use terminal value critic to estimate the long-term performance
                        real_obs = {k: v[[id]] for k, v in terminal_obs.items()}
                        if self.obs_rms is not None:
                            real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}
                        next_values[i + 1, id] = self.critic_target(real_obs['obs']).squeeze(-1)

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError

            # compute actor loss
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew
            if i < self.horizon_len - 1:
                a_loss = -rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                actor_loss = actor_loss + a_loss.sum()
            else:
                # terminate all envs at the end of optimization iteration
                a_loss = -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                actor_loss = actor_loss + a_loss.sum()

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon_len - 1:
                    self.done_mask[i] = done.clone().to(dtype=torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode metrics
            with torch.no_grad():
                if len(done_env_ids) > 0:
                    done_env_ids = done_env_ids.detach().cpu()
                    self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                    self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                    self.episode_discounted_rewards_tracker.update(self.episode_discounted_rewards[done_env_ids])

                    for done_env_id in done_env_ids:
                        if self.episode_rewards[done_env_id] > 1e6 or self.episode_rewards[done_env_id] < -1e6:
                            print('ep_rewards error')
                            raise ValueError
                        self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                        self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
                        self.episode_discounted_rewards_hist.append(self.episode_discounted_rewards[done_env_id].item())
                        self.episode_rewards[done_env_id] = 0.0
                        self.episode_lengths[done_env_id] = 0
                        self.episode_discounted_rewards[done_env_id] = 0.0
                        self.episode_gamma[done_env_id] = 1.0

        self.agent_steps += self.horizon_len * self.num_envs

        actor_loss /= self.horizon_len * self.num_envs
        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
        return actor_loss

    def update_critic(self, dataset):
        results = collections.defaultdict(list)
        for j in range(self.critic_iterations):
            total_critic_loss = 0.0
            B = len(dataset)

            for i in range(B):
                batch_sample = dataset[i]
                b_obs, b_target_values = batch_sample

                self.critic_optim.zero_grad()
                critic_loss = self.compute_critic_loss(b_obs, b_target_values)
                critic_loss.backward()

                # ugly fix for simulation nan problem
                for params in self.critic.parameters():
                    params.grad.nan_to_num_(0.0, 0.0, 0.0)

                if self.shac_config.truncate_grads:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.shac_config.max_grad_norm)

                self.critic_optim.step()
                total_critic_loss += critic_loss
            value_loss = (total_critic_loss / B).detach()
            results["value_loss"].append(value_loss)

        #     print(f'value iter {j+1}/{self.critic_iterations}, value_loss= {value_loss.item():7.6f}', end='\r')
        # print()
        return results

    def compute_critic_loss(self, obs, target_values):
        predicted_values = self.critic(obs['obs']).squeeze(-1)
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon_len)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                adjusted_rew = (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + adjusted_rew)
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError(self.critic_method)

    def eval(self):
        mean_episode_rewards, mean_episode_lengths, mean_episode_discounted_rewards = self.evaluate_policy(
            num_episodes=self.num_actors, deterministic=True,
        )
        print(
            f'mean ep_rewards = {mean_episode_rewards},',
            f'mean ep_lengths = {mean_episode_lengths}',
            f'mean ep_discounted_rewards = {mean_episode_discounted_rewards},',
        )

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def save(self, f):
        ckpt = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'obs_rms': self.obs_rms.state_dict() if self.shac_config.normalize_input else None,
            'ret_rms': self.ret_rms.state_dict() if self.shac_config.normalize_ret else None,
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        ckpt = torch.load(f, map_location=self.device)
        all_ckpt_keys = ('actor', 'critic', 'critic_target', 'obs_rms', 'ret_rms')
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f'Warning: ckpt skipped loading `{k}`')
                continue
            if k == 'obs_rms' and (not self.shac_config.normalize_input):
                continue
            if k == 'ret_rms' and (not self.shac_config.normalize_ret):
                continue

            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
