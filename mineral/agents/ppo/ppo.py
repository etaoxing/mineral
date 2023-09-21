import collections
import os
import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

from .experience import ExperienceBuffer
from . import models
from .utils import RewardShaper, AverageScalarMeter, RunningMeanStd, TensorboardLogger, WandbLogger


class PPO:
    def __init__(self, env, output_dir, full_config):
        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))
            dist.init_process_group('nccl', rank=self.rank, world_size=self.rank_size)
            self.device = 'cuda:' + str(self.rank)
            print(f'current rank: {self.rank} and use device {self.device}')
        else:
            self.rank = -1
            self.device = full_config.rl_device
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_dim = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.call_env_reset = full_config.task.get('call_env_reset', False)  # False for vectorized envs that auto reset
        # ---- Logging ----
        self.info_keys_video = full_config.train.get('info_keys_video', '$^')
        self.info_keys_sum = full_config.train.get('info_keys_sum', '$^')
        self.info_keys_min = full_config.train.get('info_keys_min', '$^')
        self.info_keys_max = full_config.train.get('info_keys_max', '$^')
        self.info_keys_final = full_config.train.get('info_keys_final', '$^')
        self.info_keys_scalar = full_config.train.get('info_keys_scalar', '$^')
        self.env_render = full_config.env_render
        # ---- Model ----
        self.obs_keys_cpu = full_config.train.get('obs_keys_cpu', '$^')
        self.input_keys_normalize = full_config.train.get('input_keys_normalize', '')
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
        self.obs_space = obs_space
        ModelCls = getattr(models, self.network_config.get('model_cls', 'ActorCritic'))
        self.model = ModelCls(obs_space, self.actions_dim, self.network_config)
        self.model.to(self.device)
        print(self.model, '\n')
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        self.tb_dir = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
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
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_frequency = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        self.save_video_every = self.ppo_config['save_video_every']
        self.save_video_consecutive = self.ppo_config['save_video_consecutive']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        self._episode_info = {}
        self._video_buf = collections.defaultdict(list)
        self._info_video = None
        self._info_keys_stats = collections.defaultdict(list)
        resolved_config = OmegaConf.to_container(full_config, resolve=True)
        self._writers = [
            WandbLogger(),
            TensorboardLogger(self.tb_dir, resolved_config),
        ]

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = -1
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_space, self.actions_dim, self.device, self.obs_keys_cpu,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = int(self.ppo_config['max_agent_steps'])
        self.best_rewards = -10000
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
                last_sps = (
                    self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size) \
                    / (time.time() - _last_t)
                _last_t = time.time()
                info_string = f'Epoch: {self.epoch_num} | ' \
                              f'Agent Steps: {int(self.agent_steps):,} | SPS: {all_sps:.1f} | ' \
                              f'Last SPS: {last_sps:.1f} | ' \
                              f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                              f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                              f'Current Best: {self.best_rewards:.2f}'
                print(info_string)

                mean_rewards = self.episode_rewards.get_mean()
                metrics = {
                    'runtime/time/env': self.data_collect_time,
                    'runtime/time/rl': self.rl_train_time,
                    'runtime/time/total': total_time,
                    'runtime/sps/env': self.agent_steps / self.data_collect_time,
                    'runtime/sps/rl': self.agent_steps / self.rl_train_time,
                    'runtime/sps/total': all_sps,
                    'metrics/episode_rewards': mean_rewards,
                    'metrics/episode_lengths': self.episode_lengths.get_mean(),
                }
                summary = self.summary_stats(self.agent_steps, train_result, metrics)
                [w(summary) for w in self._writers]

                ckpt_name = f'ep={self.epoch_num}_step={int(self.agent_steps // 1e6):04}m_reward={mean_rewards:.2f}'
                if self.save_frequency > 0:
                    if (self.epoch_num % self.save_frequency == 0) and (mean_rewards <= self.best_rewards):
                        self.save(os.path.join(self.ckpt_dir, ckpt_name))
                    self.save(os.path.join(self.ckpt_dir, f'last'))

                if mean_rewards > self.best_rewards:
                    print(f'saving current best_reward={mean_rewards:.2f}')
                    # remove previous best file
                    prev_best_ckpt = os.path.join(self.ckpt_dir, f'best_reward={self.best_rewards:.2f}.pth')
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.ckpt_dir, f'best_reward={mean_rewards:.2f}'))

        print('max steps achieved')

    def test(self, video_fn='eval'):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            info['reward'] = r

    def _convert_obs(self):
        obs = {}
        for k, v in self.obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = torch.tensor(v, device=self.device if not re.match(self.obs_keys_cpu, k) else 'cpu')
            else:
                # assert isinstance(v, torch.Tensor)
                obs[k] = v
        return obs

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
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

                a_loss, clip_frac = actor_loss(old_action_log_probs, action_log_probs, advantage, self.e_clip, self.use_smooth_clamp)
                c_loss = critic_loss(value_preds, values, self.e_clip, returns, self.clip_value_loss)
                b_loss = bounds_loss(mu, self.bounds_type)

                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef

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
                                all_grads[offset: offset + param.numel()].view_as(param.grad.data) / self.rank_size
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

        self.rl_train_time += (time.time() - _t)
        return train_result

    def play_steps(self):
        for n in range(self.horizon_length):
            if self.call_env_reset:
                if any(self.dones):
                    done_indices = self.dones.nonzero(as_tuple=False)
                    env_indices = done_indices.squeeze(-1).cpu().numpy().tolist()
                    obs_reset = self.env.reset_idx(env_indices)
                    for k, v in obs_reset.items():
                        self.obs[k][env_indices] = v

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

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            self._save_info(infos)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        if self.save_video_every > 0:
            # saved video steps depends on horizon_length in play_steps()
            if (self.epoch_num % self.save_video_every) == (self.save_video_consecutive - 1):
                self._info_video = {f'video/{k}': np.concatenate(v, 1) for k, v in self._video_buf.items()}
                self._video_buf = collections.defaultdict(list)

        obs = self._convert_obs()
        res_dict = self.model_act(obs)
        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size) if not self.multi_gpu else self.agent_steps + self.batch_size * self.rank_size
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

    def _reshape_env_render(self, k, v):
        if len(v.shape) == 3:  # H, W, C
            v = v[None][None]  # -> B, T, H, W, C
        elif len(v.shape) == 4:  # B, H, W, C
            v = v[:, None, ...]  # -> B, T, H, W, C
        elif len(v.shape) == 5:  # B, T, H, W, C
            pass
        else:
            raise RuntimeError(f'Unsupported {k} shape {v.shape}')
        return v

    def _save_info(self, infos):
        for k, v in infos.items():
            if re.match(self.info_keys_scalar, k):
                # assert isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                self.extra_info[k] = v.item()

        if self.save_video_every > 0:
            if (self.epoch_num % self.save_video_every < self.save_video_consecutive):
                if self.env_render:
                    v = self.env.render(mode='rgb_array')
                    v = self._reshape_env_render('env_render', v)
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    self._video_buf['env_render'].append(v)

                for k, v in self.obs.items():
                    if re.match(self.info_keys_video, k):
                        v = self._reshape_env_render(k, v)
                        if isinstance(v, torch.Tensor):
                            v = v.cpu().numpy()
                        self._video_buf[k].append(v)

        ep = self._episode_info
        done_indices = self.dones.nonzero(as_tuple=False)
        done_indices = done_indices.squeeze(-1).cpu().numpy().tolist()
        for k, v in self.obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            if re.match(self.info_keys_sum, k):
                _k = f'{k}_sum'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] += v
            if re.match(self.info_keys_min, k):
                _k = f'{k}_min'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] = torch.fmin(ep[_k], v)
            if re.match(self.info_keys_max, k):
                _k = f'{k}_max'
                if _k not in ep:
                    ep[_k] = v.clone()
                else:
                    ep[_k] = torch.fmax(ep[_k], v)
            if re.match(self.info_keys_final, k):
                _k = f'{k}_final'
                if _k not in ep:
                    ep[_k] = torch.zeros_like(v)
                ep[_k][done_indices] = v[done_indices]

        if len(done_indices) > 0:
            for k in ep.keys():
                v = ep[k][done_indices].cpu().numpy()
                self._info_keys_stats[k].append(v)
                if 'sum' in k or 'final' in k:
                    ep[k][done_indices] = 0
                else:
                    ep[k][done_indices] = torch.nan

    def summary_stats(self, step, train_result, metrics):
        log_dict = {
            'train/loss/total': torch.mean(torch.stack(train_result['loss'])).item(),
            'train/loss/actor': torch.mean(torch.stack(train_result['a_loss'])).item(),
            'train/loss/bounds': torch.mean(torch.stack(train_result['b_loss'])).item(),
            'train/loss/critic': torch.mean(torch.stack(train_result['c_loss'])).item(),
            'train/loss/entropy': torch.mean(torch.stack(train_result['entropy'])).item(),
            'train/avg_kl': torch.mean(torch.stack(train_result['avg_kl'])).item(),
            'train/clip_frac': torch.mean(torch.stack(train_result['clip_frac'])).item(),
            'train/last_lr': self.last_lr,
            'train/e_clip': self.e_clip,
            'train/epoch': self.epoch_num,
        }

        summary = {**log_dict, **metrics}

        for k, v in self._info_keys_stats.items():
            v = np.concatenate(v, 0)
            summary[f'episode/{k}'] = np.nanmean(v).item()
        self._info_keys_stats.clear()

        for k, v in self.extra_info.items():
            summary[f'extras/{k}'] = v

        if self._info_video is not None:
            summary.update(self._info_video)
            self._info_video = None

        summary = tuple([(step, k, v) for k, v in summary.items()])
        return summary


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
        value_pred_clipped = value_preds + \
            (values - value_preds).clamp(-e_clip, e_clip)
        value_losses = (values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        c_loss = torch.max(value_losses, value_losses_clipped)
    else:
        c_loss = (values - returns) ** 2
    return c_loss


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
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)


def adjust_learning_rate_cos(init_lr, epoch, mini_epochs, agent_steps, max_agent_steps):
    lr = init_lr * 0.5 * (
        1. + math.cos(
            math.pi * (agent_steps + epoch / mini_epochs) / max_agent_steps))
    return lr
