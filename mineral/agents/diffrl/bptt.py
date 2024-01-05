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
from ..agent import Agent
from . import models
from .utils import grad_norm


class BPTT(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.bptt_config = full_cfg.agent.bptt
        self.num_actors = self.bptt_config.num_actors
        self.max_agent_steps = int(self.bptt_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length

        # --- BPTT parameters ---
        self.gamma = self.bptt_config.get('gamma', 0.99)

        self.horizon_len = self.bptt_config.horizon_len
        self.max_epochs = self.bptt_config.max_epochs

        # --- Normalizers ---
        rms_config = dict(eps=1e-5, correction=0, initial_count=1e-4, dtype=torch.float64)  # unbiased=False -> correction=0
        if self.bptt_config.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Model ---
        obs_dim = self.obs_space['obs']
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
        assert obs_dim == self.env.num_obs
        assert self.action_dim == self.env.num_actions

        ActorCls = getattr(models, self.network_config.actor)
        self.actor = ActorCls(obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.actor.to(self.device)
        print('Actor:', self.actor, '\n')

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.bptt_config.optim_type)
        self.actor_optim = OptimCls(
            self.actor.parameters(),
            **self.bptt_config.get("actor_optim_kwargs", {}),
        )
        print('Actor Optim:', self.actor_optim, '\n')
        self.actor_lr = self.actor_optim.defaults["lr"]

        # --- Replay Buffer ---
        self.reward_shaper = RewardShaper(**self.bptt_config.reward_shaper)

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

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

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
            if self.bptt_config.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(self.epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
            elif self.bptt_config.lr_schedule == 'constant':
                lr = self.actor_lr
            else:
                raise NotImplementedError(self.bptt_config.lr_schedule)

            # train actor
            self.timer.start("train/update_actor")
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            # gather train metrics
            results = {**actor_results}
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "lr": lr})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # timing metrics
            timings_total_names = ("train/update_actor", "train/make_critic_dataset", "train/update_critic")
            timings = self.timer.stats(step=self.agent_steps, total_names=timings_total_names, reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}
            metrics.update(timing_metrics)

            # episode metrics
            if len(self.episode_rewards_hist) > 0:
                mean_episode_rewards = self.episode_rewards_tracker.mean()
                mean_episode_lengths = self.episode_lengths_tracker.mean()
                mean_episode_discounted_rewards = self.episode_discounted_rewards_tracker.mean()

                episode_metrics = {
                    "train_scores/episode_rewards": mean_episode_rewards,
                    "train_scores/episode_lengths": mean_episode_lengths,
                    "train_scores/episode_discounted_rewards": mean_episode_discounted_rewards,
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
                    f'SPS: {timings["lastrate"]:.2f} |',  # actually totalrate since we don't reset the timer
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    f'Stats:',
                    f'ep_rewards {mean_episode_rewards:.2f},',
                    f'ep_lenths {mean_episode_lengths:.2f},',
                    f'ep_discounted_rewards {mean_episode_discounted_rewards:.2f},',
                    f'actor_loss {metrics["train_stats/actor_loss"]:.4f},',
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
                if self.bptt_config.truncate_grads:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.bptt_config.max_grad_norm)
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                if torch.isnan(grad_norm_before_clip):
                    print('NaN gradient')
                    raise ValueError
                    # # JIE
                    # print('here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NaN gradient')
                    # import IPython
                    # IPython.embed()
                    # for params in self.actor.parameters():
                    #     params.grad.zero_()

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

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

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

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            # compute actor loss
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew
            if i < self.horizon_len - 1:
                a_loss = -rew_acc[i + 1, done_env_ids]
                actor_loss = actor_loss + a_loss.sum()
            else:
                # terminate all envs at the end of optimization iteration
                a_loss = -rew_acc[i + 1, :]
                actor_loss = actor_loss + a_loss.sum()

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

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
        return actor_loss

    def eval(self):
        mean_episode_rewards, mean_episode_lengths, mean_episode_discounted_rewards = self.evaluate_policy(
            num_episodes=self.num_actors,
            deterministic=True,
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
            'obs_rms': self.obs_rms.state_dict() if self.bptt_config.normalize_input else None,
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        all_ckpt_keys = ('actor', 'obs_rms')
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f'Warning: ckpt skipped loading `{k}`')
                continue
            if k == 'obs_rms' and (not self.bptt_config.normalize_input):
                continue

            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
