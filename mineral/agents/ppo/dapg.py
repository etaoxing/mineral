import re

import torch


class DAPGMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---- DAPG Param ----
        self.dapg_config = self.ppo_config.get('dapg', None)
        if self.dapg_config is not None:
            self.dapg_damping = self.dapg_config.get('damping', 0.995)
            self.init_dapg_lambda = self.dapg_config.get('lambda', 0.1)
            self.dapg_lambda = self.init_dapg_lambda

        self.demo_dataloader = None
        self.demo_iter = None

    def dataloader(self, dataset, split='train'):
        sampler = None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            # shuffle=False,
            num_workers=0,
            # num_workers=self.dapg_config.num_workers,
            shuffle=(sampler is None),
            # worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=True,
        )
        return loader

    def update_dapg(self):
        assert self.dapg_config is not None
        try:
            demo_batch = next(self.demo_iter)
        except:
            if self.demo_dataloader is None:
                self.demo_dataloader = self.dataloader(self.datasets['train'])
            self.demo_iter = iter(self.demo_dataloader)
            demo_batch = next(self.demo_iter)

        demo_obs_dict, demo_action, demo_reward, demo_done, demo_info = demo_batch

        demo_obs_dict = {
            k: v.to(device='cpu' if re.match(self.obs_keys_cpu, k) else self.device) for k, v in demo_obs_dict.items()
        }
        demo_action = demo_action.to(self.device)
        demo_reward, demo_done = demo_reward.to(self.device), demo_done.to(self.device)

        B, T = demo_done.shape[:2]
        demo_obs_dict = {k: v.reshape(B * T, *v.shape[2:]) for k, v in demo_obs_dict.items()}
        demo_action = demo_action.reshape(B * T, -1)

        demo_reward = self.reward_shaper(demo_reward)

        demo_obs_dict = {k: self.obs_rms[k].normalize(v) for k, v in demo_obs_dict.items()}
        demo_batch_dict = {
            'prev_actions': demo_action,
            **demo_obs_dict,
        }
        demo_model_out = self.model(demo_batch_dict)
        demo_action_log_probs = demo_model_out['prev_neglogp']
        demo_nll_loss = torch.mean(demo_action_log_probs)
        demo_actor_loss = self.dapg_lambda * demo_nll_loss

        return demo_actor_loss, demo_nll_loss

    def update_dapg_lambda(self):
        self.dapg_lambda = self.init_dapg_lambda * (self.dapg_damping**self.epoch)
