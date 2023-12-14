import collections
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ..actorcritic_base import ActorCriticBase
from . import models


class BC(ActorCriticBase):
    def __init__(self, env, output_dir, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.bc_config = full_cfg.agent.bc
        self.num_actors = self.bc_config.num_actors
        self.max_epochs = int(self.bc_config.max_epochs)
        super().__init__(env, output_dir, full_cfg, **kwargs)

        encoder, encoder_kwargs = self.network_config.get('encoder', None), self.network_config.get('encoder_kwargs', None)
        ModelCls = getattr(models, self.network_config.get('model'))
        model_kwargs = self.network_config.get('model_kwargs', {})
        self.model = ModelCls(self.obs_space, self.action_dim, encoder=encoder, encoder_kwargs=encoder_kwargs, **model_kwargs)
        self.model = self.model.to(self.device)
        print(self.model, '\n')

        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.normalize_keys_rms, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        OptimCls = getattr(torch.optim, self.bc_config.optim_type)
        self.optim = OptimCls(
            self.model.parameters(),
            **self.bc_config.get('optim_kwargs', {}),
        )
        print(self.optim, '\n')

        self.reward_shaper = RewardShaper(**self.bc_config.reward_shaper)

    def dataloader(self, dataset, split='train'):
        sampler = None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bc_config.batch_size,
            # shuffle=False,
            # num_workers=0,
            num_workers=self.bc_config.num_workers,
            shuffle=(sampler is None),
            # worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=True,
        )
        return loader

    def actor(self, obs, **kwargs):
        if self.normalize_input:
            # TODO: make this work with (B, T, ...) inputs
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

        model_out = self.model(obs, **kwargs)
        if isinstance(model_out, dict):
            mu, sigma, distr = model_out['mu'], model_out['sigma'], model_out['distr']
        else:
            mu, sigma, distr = model_out
        return mu, sigma, distr

    def train(self):
        assert self.datasets is not None
        self.train_dataloader = self.dataloader(self.datasets['train'], split='train')
        while self.epoch < self.max_epochs:
            self.epoch += 1
            self.set_train()

            for i, batch in enumerate(self.train_dataloader):
                self.mini_epoch += 1
                train_result = self.update_model(batch)
                metrics = {
                    "train/epoch": self.epoch,
                    "train/mini_epoch": self.mini_epoch,
                    **train_result,
                }
                self.writer.add(self.mini_epoch, metrics)
                self.writer.write()

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(metrics)

            if self.ckpt_every > 0 and (self.epoch + 1) % self.ckpt_every == 0:
                ckpt_file = os.path.join(self.ckpt_dir, f'epoch={self.epoch}.pth')
                self.save(ckpt_file)
                print(f'Saved ckpt {ckpt_file}')

            if self.eval_every > 0 and (self.epoch + 1) % self.eval_every == 0:
                self.eval()

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def update_model(self, batch):
        train_result = collections.defaultdict(list)

        obs, action, reward, done, info = batch

        obs = {k: v.to(device='cpu' if re.match(self.obs_keys_cpu, k) else self.device) for k, v in obs.items()}
        action, reward, done = action.to(self.device), reward.to(self.device), done.to(self.device)

        B, T = done.shape[:2]
        obs = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs.items()}
        action = action.reshape(B * T, -1)

        reward = self.reward_shaper(reward)
        mu, std, distr = self.actor(obs)

        l1_loss = F.l1_loss(mu, action)
        mse_loss = F.mse_loss(mu, action)
        nll_loss = -distr.log_prob(action).mean()

        loss = nll_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.bc_config.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters=self.optim.param_groups[0]["params"],
                max_norm=self.bc_config.max_grad_norm,
            )
        else:
            grad_norm = None
        self.optim.step()

        train_result["loss/total"].append(loss)
        train_result["loss/mse"].append(mse_loss)
        train_result["loss/nle"].append(nll_loss)
        train_result["loss/l1"].append(l1_loss)
        if grad_norm is not None:
            train_result["grad_norm/all"].append(grad_norm)

        train_result = {k: torch.stack(v).mean().item() for k, v in train_result.items()}
        return train_result

    def eval(self):
        self.set_eval()
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        self.metrics.episode_rewards.reset()
        self.metrics.episode_lengths.reset()

        max_steps = self.env.max_episode_length
        eval_episodes = 0
        while eval_episodes < self.metrics.tracker_len:
            trajectory, steps = self.explore_env(self.env, max_steps, random=False, sample=True)
            eval_episodes += self.num_actors

        eval_metrics = {
            "metrics/episode_rewards": self.metrics.episode_rewards.mean(),
            "metrics/episode_lengths": self.metrics.episode_lengths.mean(),
        }
        eval_metrics = self.metrics.result(eval_metrics)
        self.writer.add(self.mini_epoch, eval_metrics)
        self.writer.write()

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
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self.env.reset_idx(done_indices)
                    obs_reset = self._convert_obs(obs_reset)
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(self, self.obs, sample=sample)

            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)
            rewards, self.dones = torch.as_tensor(rewards, device=self.device), torch.as_tensor(dones, device=self.device)

            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update_tracker(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            # if self.bc_config.handle_timeout:
            #     dones = handle_timeout(dones, infos)
            for k, v in self.obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_rewards[:, i] = rewards
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v
            self.obs = next_obs

        self.metrics.flush_video_buf(self.epoch)

        # traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = None
        return data, timesteps * self.num_actors

    def get_actions(self, obs, sample=True):
        mu, sigma, distr = self.actor(obs)
        actions = distr.sample() if sample else mu
        return actions

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.obs_rms.train()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.obs_rms.eval()

    def save(self, f):
        ckpt = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epoch,
            'mini_epoch': self.mini_epoch,
        }
        if self.normalize_input:
            ckpt['obs_rms'] = self.obs_rms.state_dict()
        torch.save(ckpt, f)

    def load(self, f):
        ckpt = torch.load(f, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if self.normalize_input:
            self.obs_rms.load_state_dict(ckpt['obs_rms'])

        try:
            self.optim.load_state_dict(ckpt['optim'])
            self.epoch = ckpt['epoch']
            self.mini_epoch = ckpt['mini_epoch']
        except KeyError:
            print('Did not find one of {optim, epoch, mini_epoch} in checkpoint.')
