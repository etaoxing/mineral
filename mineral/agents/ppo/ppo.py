import collections
import os
import re

import torch
import torch.nn as nn

from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ..agent import Agent
from . import models
from .dapg import DAPGMixin
from .experience import ExperienceBuffer
from .utils import AdaptiveScheduler, LinearScheduler, adjust_learning_rate_cos


class PPO(DAPGMixin, Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.ppo_config = full_cfg.agent.ppo
        self.num_actors = self.ppo_config.num_actors
        self.max_agent_steps = int(self.ppo_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        # ---- Normalizers ----
        rms_config = dict(eps=1e-5, with_clamp=True, initial_count=1, dtype=torch.float64)
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
        print('obs_rms:', self.obs_rms)

        self.value_rms = normalizers.RunningMeanStd((1,), **rms_config).to(self.device)

        # ---- Model ----
        encoder, encoder_kwargs = self.network_config.get('encoder', None), self.network_config.get('encoder_kwargs', None)
        ModelCls = getattr(models, self.network_config.get('actor_critic', 'ActorCritic'))
        model_kwargs = self.network_config.get('actor_critic_kwargs', {})
        self.model = ModelCls(self.obs_space, self.action_dim, encoder=encoder, encoder_kwargs=encoder_kwargs, **model_kwargs)
        self.model.to(self.device)
        print(self.model, '\n')

        # ---- Optim ----
        optim_kwargs = self.ppo_config.get('optim_kwargs', {})
        learning_rate = optim_kwargs.get('lr', 3e-4)
        self.init_lr = float(learning_rate)
        self.last_lr = float(learning_rate)
        OptimCls = getattr(torch.optim, self.ppo_config.optim_type)
        self.optim = OptimCls(self.model.parameters(), **optim_kwargs)
        print(self.optim, '\n')

        # ---- PPO Train Params ----
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
        self.max_grad_norm = self.ppo_config['max_grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_value = self.ppo_config['normalize_value']

        # ---- PPO Collect Params ----
        self.horizon_len = self.ppo_config['horizon_len']
        self.batch_size = self.horizon_len * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_cfg.test

        # ---- LR Scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(self.init_lr, self.ppo_config['max_agent_steps'])

        # --- Training ---
        self.obs, self.dones = None, None
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_len,
            self.batch_size,
            self.minibatch_size,
            self.obs_space,
            self.action_dim,
            self.device,
            self.cpu_obs_keys,
        )
        self.reward_shaper = RewardShaper(**self.ppo_config['reward_shaper'])

        # --- Timing ---
        self.timer = Timer()
        self.timer.wrap('agent', self, ['train_epoch', 'play_steps'])
        self.timer.wrap('env', self.env, ['step'])

        # --- Multi-GPU ---
        if self.multi_gpu:
            self.model = self.accelerator.prepare(self.model)
            self.optim = self.accelerator.prepare(self.optim)
            # TODO: prepare scheduler
            if self.normalize_input:
                self.obs_rms = self.accelerator.prepare(self.obs_rms)
            if self.normalize_value:
                self.value_rms = self.accelerator.prepare(self.value_rms)

    def model_act(self, obs_dict):
        if self.normalize_input:
            obs_dict = {k: self.obs_rms[k].normalize(v) for k, v in obs_dict.items()}
        model_out = self.model.act(obs_dict)
        if self.normalize_value:
            model_out['values'] = self.value_rms.unnormalize(model_out['values'])
        return model_out

    def play_steps(self):
        for n in range(self.horizon_len):
            if not self.env_autoresets:
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self.env.reset_idx(done_indices)
                    obs_reset = self._convert_obs(obs_reset)
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            model_out = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs)
            for k in ['actions', 'neglogp', 'values', 'mu', 'sigma']:
                self.storage.update_data(k, n, model_out[k])
            # do env step
            actions = torch.clamp(model_out['actions'], -1.0, 1.0)
            obs, r, self.dones, infos = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            r, self.dones = torch.tensor(r, device=self.device), torch.tensor(self.dones, device=self.device)
            rewards = r.reshape(-1, 1)

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_shaper(rewards.clone())
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = torch.tensor(infos['time_outs'], device=self.device)
                time_outs = time_outs.reshape(-1, 1)
                shaped_rewards += self.gamma * model_out['values'] * time_outs.float()
            self.storage.update_data('rewards', n, shaped_rewards)

            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards.squeeze(-1), done_indices, infos)
        self.metrics.flush_video_buf(self.epoch)

        model_out = self.model_act(obs)
        last_values = model_out['values']

        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        values = self.storage.data_dict['values']
        returns = self.storage.data_dict['returns']
        if self.normalize_value:
            self.value_rms.train()
            self.value_rms.update(values)
            values = self.value_rms.normalize(values)
            self.value_rms.update(returns)
            returns = self.value_rms.normalize(returns)
            self.value_rms.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1

            self.set_eval()
            self.play_steps()
            self.agent_steps += self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

            self.set_train()
            results = self.train_epoch()
            self.storage.data_dict = None

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                # gather train metrics
                metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
                metrics.update({k: torch.mean(torch.cat(results[k])).item() for k in ["mu", "sigma"]})  # distributions
                metrics.update(
                    {'epoch': self.epoch, 'mini_epoch': self.mini_epoch, 'last_lr': self.last_lr, 'e_clip': self.e_clip}
                )
                metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

                # timing metrics
                timings_total_names = ('agent.play_steps', 'agent.train_epoch')
                timings = self.timer.stats(step=self.agent_steps, total_names=timings_total_names)
                timing_metrics = {f'train_timings/{k}': v for k, v in timings.items()}
                metrics.update(timing_metrics)

                # episode metrics
                episode_metrics = {
                    'train_scores/episode_rewards': self.metrics.episode_rewards.mean(),
                    'train_scores/episode_lengths': self.metrics.episode_lengths.mean(),
                }
                metrics.update(episode_metrics)
                metrics = self.metrics.result(metrics)

                self.writer.add(self.agent_steps, metrics)
                self.writer.write()

                self._checkpoint_save(metrics['train_scores/episode_rewards'])

                if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                    print(
                        f'Epoch: {self.epoch} |',
                        f'Agent Steps: {int(self.agent_steps):,} |',
                        f'SPS: {timings["totalrate"]:.2f} |',
                        f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                        f'Stats:',
                        f'last_sps {timings["lastrate"]:.2f},'
                        f'collectEnv_time {timings["agent.play_steps/total"] / 60:.1f} min,',
                        f'trainRL_time {timings["agent.train_epoch/total"] / 60:.1f} min,',
                        f'\b\b |',
                    )

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def train_epoch(self):
        results = collections.defaultdict(list)
        for mini_ep in range(0, self.mini_epochs):
            self.mini_epoch += 1
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs_dict = self.storage[i]
                if not isinstance(obs_dict, dict):
                    obs_dict = {'obs': obs_dict}

                if self.normalize_input:
                    input_dict = {}
                    for k, v in obs_dict.items():
                        self.obs_rms[k].update(v)
                        input_dict[k] = self.obs_rms[k].normalize(v)
                else:
                    input_dict = obs_dict
                batch_dict = {
                    'prev_actions': actions,
                    **input_dict,
                }

                model_out = self.model(batch_dict)
                action_log_probs = model_out['prev_neglogp']
                values = model_out['values']
                entropy = model_out['entropy']
                mu = model_out['mu']
                sigma = model_out['sigma']

                a_loss, clip_frac = actor_loss(
                    old_action_log_probs, action_log_probs, advantage, self.e_clip, self.use_smooth_clamp
                )
                c_loss, explained_var = critic_loss(value_preds, values, self.e_clip, returns, self.clip_value_loss)
                b_loss = bounds_loss(mu, self.bounds_type)

                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                if self.dapg_config is not None:
                    demo_actor_loss, demo_nll_loss = self.update_dapg()
                    loss += demo_actor_loss

                self.optim.zero_grad()
                loss.backward() if not self.multi_gpu else self.accelerator.backward(loss)

                if self.truncate_grads:
                    if not self.multi_gpu:
                        grad_norm_all = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        assert self.accelerator.sync_gradients
                        grad_norm_all = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                if self.multi_gpu:
                    metrics = (kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma)
                    metrics = self.accelerator.gather_for_metrics(metrics)
                    kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma = metrics

                    if self.dapg_config is not None:
                        demo_actor_loss, demo_nll_loss = self.accelerator.gather_for_metrics((demo_actor_loss, demo_nll_loss))

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())
                ep_kls.append(kl_dist)

                results['loss/total'].append(loss)
                results['loss/actor'].append(a_loss)
                results['loss/critic'].append(c_loss)
                results['loss/bounds'].append(b_loss)
                results['loss/entropy'].append(entropy)
                results['clip_frac'].append(clip_frac)
                results['explained_var'].append(explained_var)
                results['mu'].append(mu.detach())
                results['sigma'].append(sigma.detach())
                if self.truncate_grads:
                    results['grad_norm/all'].append(grad_norm_all)

                if self.dapg_config is not None:
                    results['dapg/demo_nll_loss'].append(demo_nll_loss)
                    results['dapg/demo_actor_loss'].append(demo_actor_loss)
                    results['dapg/lambda'].append(torch.tensor(self.dapg_lambda))
                    self.update_dapg_lambda()

            avg_kl = torch.mean(torch.stack(ep_kls))
            results['avg_kl'].append(avg_kl)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, avg_kl.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = adjust_learning_rate_cos(
                    self.init_lr, mini_ep, self.mini_epochs, self.agent_steps, self.max_agent_steps
                )

            for param_group in self.optim.param_groups:
                param_group['lr'] = self.last_lr

        if self.lr_schedule == 'linear':
            self.last_lr = self.scheduler.update(self.agent_steps)

        return results

    def eval(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            if self.normalize_input:
                obs_dict = {k: self.obs_rms[k].normalize(v) for k, v in obs_dict.items()}
            mu = self.model.act(obs_dict, sample=False)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            info['reward'] = r

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.obs_rms.train()
        if self.normalize_value:
            self.value_rms.train()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.obs_rms.eval()
        if self.normalize_value:
            self.value_rms.eval()

    def save(self, f):
        ckpt = {
            'epoch': self.epoch,
            'mini_epoch': self.mini_epoch,
            'agent_steps': self.agent_steps,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'obs_rms': self.obs_rms.state_dict() if self.normalize_input else None,
            'value_rms': self.value_rms.state_dict() if self.normalize_value else None,
        }
        torch.save(ckpt, f)
        # TODO: accelerator.save

    def load(self, f, ckpt_keys=''):
        all_ckpt_keys = ('epoch', 'mini_epoch', 'agent_steps', 'model', 'optim', 'obs_rms', 'value_rms')
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f'Warning: ckpt skipped loading `{k}`')
                continue
            if k == 'obs_rms' and (not self.normalize_input):
                continue
            if k == 'value_rms' and (not self.normalize_value):
                continue

            if hasattr(getattr(self, k), 'load_state_dict'):
                # TODO: accelerator.load
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])


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
