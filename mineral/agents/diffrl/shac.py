# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ..actorcritic_base import ActorCriticBase
from . import models
from .utils import AverageMeter, CriticDataset, grad_norm


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

        self._training = True
        if self._training:
            # save interval
            self.save_interval = self.shac_config.get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (
                self.shac_config['player'].get('determenistic', False)
                or self.shac_config['player'].get('deterministic', False)
            )
            self.horizon_len = self.env.episode_length

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

        # counting variables
        self.epoch = 0
        self.agent_steps = 0

        # loss variables
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=int)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

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

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

            if self.ret_rms is not None:
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

        for i in range(self.horizon_len):
            # collect data for critic training
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            actions = self.actor(obs['obs'], deterministic=deterministic)

            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))
            obs = self._convert_obs(obs)

            with torch.no_grad():
                raw_rew = rew.clone()

            # scale the reward
            rew = self.reward_shaper(rew)

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

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            next_values[i + 1] = self.critic_target(obs['obs']).squeeze(-1)

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
                    elif self.episode_length[id] < self.max_episode_length:  # early termination
                        next_values[i + 1, id] = 0.0
                    else:  # otherwise, use terminal value critic to estimate the long-term performance
                        real_obs = {k: v[[id]] for k, v in terminal_obs.items()}
                        if self.obs_rms is not None:
                            real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}
                        next_values[i + 1, id] = self.critic_target(real_obs['obs']).squeeze(-1)

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError

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

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    done_env_ids = done_env_ids.detach().cpu()
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6:
                            print('ep loss error')
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.0
                        self.episode_discounted_loss[done_env_id] = 0.0
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.0

        actor_loss /= self.horizon_len * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()

        self.agent_steps += self.horizon_len * self.num_envs

        return actor_loss

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic=False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_length = torch.zeros(self.num_envs, dtype=int)
        episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        obs = self.env.reset()
        obs = self._convert_obs(obs)

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

            actions = self.actor(obs['obs'], deterministic=deterministic)

            obs, rew, done, _ = self.env.step(torch.tanh(actions))
            obs = self._convert_obs(obs)

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.0
                    episode_discounted_loss[done_env_id] = 0.0
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.0
                    games_cnt += 1

        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

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

    def compute_critic_loss(self, obs, target_values):
        predicted_values = self.critic(obs['obs']).squeeze(-1)
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(
            num_games=num_games, deterministic=not self.stochastic_evaluation
        )
        print(
            f'mean episode loss = {mean_policy_loss},',
            f'mean discounted loss = {mean_policy_discounted_loss},',
            f'mean episode length = {mean_episode_length}',
        )

    def train(self):
        self.start_time = time.time()

        # initializations
        self.initialize_env()

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
                self.grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.shac_config.truncate_grads:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.shac_config.max_grad_norm)
                self.grad_norm_after_clip = grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.0:
                    print('NaN gradient')
                    raise ValueError

            self.timer.end("train/actor_closure/actor_loss")
            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.shac_config.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optim.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # train actor
            self.timer.start("train/update_actor")
            self.actor_optim.step(actor_closure).detach().item()
            self.timer.end("train/update_actor")

            # train critic
            # prepare dataset
            self.timer.start("train/make_critic_dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last=False)
            self.timer.end("train/make_critic_dataset")

            self.timer.start("train/update_critic")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    b_obs, b_target_values = batch_sample

                    self.critic_optim.zero_grad()
                    training_critic_loss = self.compute_critic_loss(b_obs, b_target_values)
                    training_critic_loss.backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.shac_config.truncate_grads:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.shac_config.max_grad_norm)

                    self.critic_optim.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.timer.end("train/update_critic")

            self.epoch += 1

            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            writer = self.tb_summary_writer
            writer.add_scalar('lr/iter', lr, self.epoch)
            writer.add_scalar('actor_loss/step', self.actor_loss, self.agent_steps)
            writer.add_scalar('actor_loss/iter', self.actor_loss, self.epoch)
            writer.add_scalar('value_loss/step', self.value_loss, self.agent_steps)
            writer.add_scalar('value_loss/iter', self.value_loss, self.epoch)
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save(os.path.join(self.ckpt_dir, 'best.pth'))
                    self.best_policy_loss = mean_policy_loss

                writer.add_scalar('policy_loss/step', mean_policy_loss, self.agent_steps)
                writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                writer.add_scalar('policy_loss/iter', mean_policy_loss, self.epoch)
                writer.add_scalar('rewards/step', -mean_policy_loss, self.agent_steps)
                writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                writer.add_scalar('rewards/iter', -mean_policy_loss, self.epoch)
                writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.agent_steps)
                writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.epoch)
                writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.agent_steps)
                writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.epoch)
                writer.add_scalar('episode_lengths/step', mean_episode_length, self.agent_steps)
                writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
                writer.add_scalar('episode_lengths/iter', mean_episode_length, self.epoch)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            fps = self.horizon_len * self.num_envs / (time_end_epoch - time_start_epoch)
            print(
                f'iter {self.epoch}:',
                f'ep loss {mean_policy_loss:.2f},',
                f'ep discounted loss {mean_policy_discounted_loss:.2f},',
                f'ep len {mean_episode_length:.1f},',
                f'fps total {fps:.2f},',
                f'value loss {self.value_loss:.2f},',
                f'grad norm before clip {self.grad_norm_before_clip:.2f},',
                f'grad norm after clip {self.grad_norm_after_clip:.2f},',
            )

            writer.flush()

            if self.save_interval > 0 and (self.epoch % self.save_interval == 0):
                self.save("policy_iter{}_reward{:.3f}".format(self.epoch, -mean_policy_loss))

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1.0 - alpha) * param.data)

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.logdir, 'episode_loss_his.npy'), 'wb'), self.episode_loss_his)
        np.save(open(os.path.join(self.logdir, 'episode_discounted_loss_his.npy'), 'wb'), self.episode_discounted_loss_his)
        np.save(open(os.path.join(self.logdir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # evaluate the final policy's performance
        self.run(self.num_envs)

    def play(self, cfg):
        self.load(cfg['params']['general']['checkpoint'])
        self.run(self.shac_config['player']['games_num'])

    def save(self, f):
        return
        ckpt = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'obs_rms': self.obs_rms.state_dict() if self.shac_config.normalize_input else None,
            'ret_rms': self.ret_rms.state_dict() if self.shac_config.normalize_ret else None,
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        return
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
