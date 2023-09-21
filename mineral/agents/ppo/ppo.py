import collections
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from ..actorcritic_base import ActorCriticBase
from ..nets import RunningMeanStd
from . import models
from .experience import ExperienceBuffer
from .utils import AdaptiveScheduler, LinearScheduler, RewardShaper, adjust_learning_rate_cos


class PPO(ActorCriticBase):
    def __init__(self, env, output_dir, full_cfg):
        self.network_config = full_cfg.agent.network
        self.ppo_config = full_cfg.agent.ppo
        self.num_actors = self.ppo_config['num_actors']
        super().__init__(env, output_dir, full_cfg)

        # --- Multi GPU ---
        self.multi_gpu = full_cfg.agent.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))
            dist.init_process_group('nccl', rank=self.rank, world_size=self.rank_size)
            self.device = 'cuda:' + str(self.rank)
            print(f'current rank: {self.rank} and use device {self.device}')
        # ---- Environment ----
        action_space = self.env.action_space
        self.actions_dim = action_space.shape[0]
        self.observation_space = self.env.observation_space
        self.call_env_reset = full_cfg.task.get('call_env_reset', False)  # False for vectorized envs that auto reset
        # ---- Input Normalization ----
        self.obs_keys_cpu = full_cfg.agent.get('obs_keys_cpu', '$^')
        self.input_keys_normalize = full_cfg.agent.get('input_keys_normalize', '')
        try:
            obs_space = {k: v.shape for k, v in self.observation_space.spaces.items()}
        except:
            obs_space = {'obs': self.observation_space.shape}
        if self.ppo_config['normalize_input']:
            self.running_mean_std = {
                k: RunningMeanStd(v).to(self.device) if re.match(self.input_keys_normalize, k) else nn.Identity()
                for k, v in obs_space.items()
            }
            self.running_mean_std = nn.ModuleDict(self.running_mean_std)
            print('RunningMeanStd:', self.running_mean_std)
        # ---- Model ----
        self.obs_space = obs_space
        ModelCls = getattr(models, self.network_config.get('model_cls', 'ActorCritic'))
        self.model = ModelCls(obs_space, self.actions_dim, self.network_config)
        self.model.to(self.device)
        print(self.model, '\n')
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.init_lr, eps=self.ppo_config['adam_eps'])
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.use_smooth_clamp = self.ppo_config['use_smooth_clamp']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.bounds_type = self.ppo_config['bounds_type']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scaler = RewardShaper(**self.ppo_config['reward_shaper'])
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_cfg.test
        # ---- Scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(self.init_lr, self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_frequency = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # --- Training ---
        self.obs = None
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_space,
            self.actions_dim,
            self.device,
            self.obs_keys_cpu,
        )
        self.best_rewards = -10000
        self.epoch_num = -1
        self.agent_steps = 0
        self.max_agent_steps = int(self.ppo_config['max_agent_steps'])
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)
        self.agent_steps = self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            train_result = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                total_time = time.time() - _t
                all_sps = self.agent_steps / total_time
                last_sps = (self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size) / (
                    time.time() - _last_t
                )
                _last_t = time.time()
                info_string = (
                    f'Epoch: {self.epoch_num} | '
                    f'Agent Steps: {int(self.agent_steps):,} | SPS: {all_sps:.1f} | '
                    f'Last SPS: {last_sps:.1f} | '
                    f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                    f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                    f'Current Best: {self.best_rewards:.2f}'
                )
                print(info_string)

                metrics = self.summary_stats(total_time, all_sps, train_result)
                self.write_metrics(self.agent_steps, metrics)

                mean_rewards = metrics['metrics/episode_rewards']
                ckpt_name = f'ep={self.epoch_num}_step={int(self.agent_steps // 1e6):04}m_reward={mean_rewards:.2f}'
                if self.save_frequency > 0:
                    if (self.epoch_num % self.save_frequency == 0) and (mean_rewards <= self.best_rewards):
                        self.save(os.path.join(self.ckpt_dir, ckpt_name))
                    self.save(os.path.join(self.ckpt_dir, f'last'))

                if mean_rewards > self.best_rewards:
                    print(f'saving current best_rewards={mean_rewards:.2f}')
                    # remove previous best file
                    prev_best_ckpt = os.path.join(self.ckpt_dir, f'best_reward={self.best_rewards:.2f}.pth')
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.ckpt_dir, f'best_reward={mean_rewards:.2f}'))

        print('max steps achieved')

    def summary_stats(self, total_time, all_sps, train_result):
        metrics = {
            'runtime/time/env': self.data_collect_time,
            'runtime/time/rl': self.rl_train_time,
            'runtime/time/total': total_time,
            'runtime/sps/env': self.agent_steps / self.data_collect_time,
            'runtime/sps/rl': self.agent_steps / self.rl_train_time,
            'runtime/sps/total': all_sps,
            'metrics/episode_rewards': self.episode_rewards.mean(),
            'metrics/episode_lengths': self.episode_lengths.mean(),
        }
        log_dict = {
            'train/loss/total': torch.mean(torch.stack(train_result['loss'])).item(),
            'train/loss/actor': torch.mean(torch.stack(train_result['a_loss'])).item(),
            'train/loss/bounds': torch.mean(torch.stack(train_result['b_loss'])).item(),
            'train/loss/critic': torch.mean(torch.stack(train_result['c_loss'])).item(),
            'train/loss/entropy': torch.mean(torch.stack(train_result['entropy'])).item(),
            'train/avg_kl': torch.mean(torch.stack(train_result['avg_kl'])).item(),
            'train/clip_frac': torch.mean(torch.stack(train_result['clip_frac'])).item(),
            'train/explained_var': torch.mean(torch.stack(train_result['explained_var'])).item(),
            'train/last_lr': self.last_lr,
            'train/e_clip': self.e_clip,
            'train/epoch': self.epoch_num,
        }
        return {**metrics, **log_dict}

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            info['reward'] = r

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        # update network
        _t = time.time()
        self.set_train()

        train_result = collections.defaultdict(list)
        for mini_ep in range(0, self.mini_epochs):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs_dict = self.storage[i]
                if not isinstance(obs_dict, dict):
                    obs_dict = {'obs': obs_dict}

                input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
                batch_dict = {
                    'prev_actions': actions,
                    **input_dict,
                }

                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                a_loss, clip_frac = actor_loss(
                    old_action_log_probs, action_log_probs, advantage, self.e_clip, self.use_smooth_clamp
                )
                c_loss, explained_var = critic_loss(value_preds, values, self.e_clip, returns, self.clip_value_loss)
                b_loss = bounds_loss(mu, self.bounds_type)

                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                            )
                            offset += param.numel()

                if self.truncate_grads:
                    grad_norm_all = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                ep_kls.append(kl_dist)
                train_result['loss'].append(loss)
                train_result['a_loss'].append(a_loss)
                train_result['c_loss'].append(c_loss)
                train_result['b_loss'].append(b_loss)
                train_result['entropy'].append(entropy)
                train_result['clip_frac'].append(clip_frac)
                train_result['explained_var'].append(explained_var)
                if self.truncate_grads:
                    train_result['grad_norm/all'].append(grad_norm_all.item())

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            avg_kl = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(avg_kl, op=dist.ReduceOp.SUM)
                avg_kl /= self.rank_size
            train_result['avg_kl'].append(avg_kl)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, avg_kl.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = adjust_learning_rate_cos(
                    self.init_lr, mini_ep, self.mini_epochs, self.agent_steps, self.max_agent_steps
                )

            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.last_lr

        self.rl_train_time += time.time() - _t
        return train_result

    def _convert_obs(self):
        obs = {}
        for k, v in self.obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = torch.tensor(v, device=self.device if not re.match(self.obs_keys_cpu, k) else 'cpu')
            else:
                # assert isinstance(v, torch.Tensor)
                obs[k] = v
        return obs

    def play_steps(self):
        for n in range(self.horizon_length):
            if self.call_env_reset:
                if any(self.dones):
                    done_indices = self.dones.nonzero(as_tuple=False)
                    done_indices = done_indices.squeeze(-1).cpu().numpy().tolist()
                    obs_reset = self.env.reset_idx(done_indices)
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            self.obs = self._convert_obs()
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs)
            for k in ['actions', 'neglogp', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, r, self.dones, infos = self.env.step(actions)
            r, self.dones = torch.tensor(r, device=self.device), torch.tensor(self.dones, device=self.device)
            rewards = r.reshape(-1, 1)

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scaler(rewards.clone())
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = torch.tensor(infos['time_outs'], device=self.device)
                time_outs = time_outs.reshape(-1, 1)
                shaped_rewards += self.gamma * res_dict['values'] * time_outs.float()
            self.storage.update_data('rewards', n, shaped_rewards)

            done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = done_indices.squeeze(-1).cpu().numpy().tolist()
            save_video = (self.save_video_every) > 0 and (self.epoch_num % self.save_video_every < self.save_video_consecutive)
            self._update_tracker(rewards.squeeze(-1), done_indices, infos, save_video=save_video)

        if self.save_video_every > 0:
            # saved video steps depends on horizon_length in play_steps()
            if (self.epoch_num % self.save_video_every) == (self.save_video_consecutive - 1):
                self._info_video = {f'video/{k}': np.concatenate(v, 1) for k, v in self._video_buf.items()}
                self._video_buf = collections.defaultdict(list)

        obs = self._convert_obs()
        res_dict = self.model_act(obs)
        last_values = res_dict['values']

        self.agent_steps = (
            (self.agent_steps + self.batch_size) if not self.multi_gpu else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, f):
        if not f:
            return
        checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, f):
        checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


def smooth_clamp(x, mi, mx):
    return 1 / (1 + torch.exp((-(x - mi) / (mx - mi) + 0.5) * 4)) * (mx - mi) + mi


def actor_loss(old_action_log_probs, action_log_probs, advantage, e_clip, use_smooth_clamp):
    clamp = smooth_clamp if use_smooth_clamp else torch.clamp
    ratio = torch.exp(old_action_log_probs - action_log_probs)
    surr1 = advantage * ratio
    surr2 = advantage * clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    a_loss = torch.max(-surr1, -surr2)

    # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/665c32170d84b4be66722eea405a1e08b6e7f761/isaacgymenvs/learning/common_agent.py#L484
    clipped = torch.abs(ratio - 1.0) > e_clip
    clip_frac = torch.mean(clipped.float()).detach()
    return a_loss, clip_frac


def critic_loss(value_preds, values, e_clip, returns, clip_value_loss):
    if clip_value_loss:
        value_pred_clipped = value_preds + (values - value_preds).clamp(-e_clip, e_clip)
        value_losses = (values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        c_loss = torch.max(value_losses, value_losses_clipped)
    else:
        c_loss = (values - returns) ** 2

    explained_var = (1 - torch.var(returns - values) / (torch.var(returns) + 1e-8)).clamp(0, 1).detach()
    return c_loss, explained_var


def bounds_loss(mu, bounds_type, soft_bound=1.1):
    if bounds_type == 'bound':
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        return b_loss
    elif bounds_type == 'reg':
        reg_loss = (mu * mu).sum(axis=-1)
        return reg_loss
    else:
        raise NotImplementedError(bounds_type)


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()
