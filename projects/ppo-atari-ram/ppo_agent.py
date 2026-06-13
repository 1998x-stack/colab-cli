"""PPO Agent with GAE — CleanRL-style clipped objective."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
import numpy as np


class PPOAgent:
    def __init__(
        self,
        network: nn.Module,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        n_minibatches: int = 4,
        device: str = "cuda",
    ):
        self.net = network.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(obs)
        return action.cpu(), log_prob.cpu(), value.cpu()

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        _, value = self.net.forward(obs)
        return value.cpu()

    @torch.no_grad()
    def compute_returns(
        self,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and Monte Carlo returns.

        All inputs shape: (num_steps, num_envs) or (num_envs,) for next_obs/next_done.
        Returns: advantages (num_steps, num_envs), returns (num_steps, num_envs).
        """
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].float()
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )

        returns = advantages + values
        return advantages, returns

    def update(self, rollouts: dict) -> Dict[str, float]:
        """PPO update over K epochs with minibatch shuffle.

        rollouts dict keys:
            obs: (num_steps, num_envs, obs_dim)
            actions: (num_steps, num_envs)
            log_probs: (num_steps, num_envs)
            advantages: (num_steps, num_envs)
            returns: (num_steps, num_envs)
            values: (num_steps, num_envs)

        Returns dict of metrics averaged over epochs.
        """
        num_steps, num_envs = rollouts["actions"].shape
        batch_size = num_steps * num_envs
        minibatch_size = batch_size // self.n_minibatches

        # Flatten
        b_obs = rollouts["obs"].reshape(-1, rollouts["obs"].shape[-1]).to(self.device)
        b_actions = rollouts["actions"].reshape(-1).to(self.device)
        b_log_probs = rollouts["log_probs"].reshape(-1).to(self.device)
        b_advantages = rollouts["advantages"].reshape(-1).to(self.device)
        b_returns = rollouts["returns"].reshape(-1).to(self.device)
        b_values = rollouts["values"].reshape(-1).to(self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        metrics_sum = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        total_updates = 0

        for _ in range(self.n_epochs):
            idxs = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]

                _, new_log_prob, entropy, new_value = self.net.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                logratio = new_log_prob - b_log_probs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                mb_advantages = b_advantages[mb_idx]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_value - b_values[mb_idx], -self.clip_coef, self.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                vf_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * vf_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics_sum["pg_loss"] += pg_loss.item()
                metrics_sum["vf_loss"] += vf_loss.item()
                metrics_sum["entropy"] += entropy_loss.item()
                metrics_sum["clip_frac"] += clip_frac.item()
                total_updates += 1

        n = max(total_updates, 1)
        return {k: v / n for k, v in metrics_sum.items()}
