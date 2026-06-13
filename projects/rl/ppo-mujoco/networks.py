"""Actor-Critic networks for MuJoCo (continuous actions, Gaussian policy)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class MLPActorCritic(nn.Module):
    """Shared-trunk MLP with Gaussian actor for continuous actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(in_dim, n_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))
        self.critic = nn.Linear(in_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=np.sqrt(2))
        orthogonal_init(self.actor_mean, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor_mean(h), self.critic(h)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(x)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


class ResMLPActorCritic(nn.Module):
    """Residual MLP with Gaussian actor for continuous actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        self.fc_in = nn.Linear(obs_dim, hidden[0])
        self.ln_in = nn.LayerNorm(hidden[0])

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden) - 1):
            block = nn.Sequential(
                nn.Linear(hidden[i], hidden[i + 1]),
                nn.LayerNorm(hidden[i + 1]),
                nn.ReLU(),
            )
            self.res_blocks.append(block)

        self.actor_mean = nn.Linear(hidden[-1], n_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))
        self.critic = nn.Linear(hidden[-1], 1)
        self.init_weights()

    def init_weights(self):
        orthogonal_init(self.fc_in, gain=np.sqrt(2))
        for block in self.res_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_mean, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.ln_in(self.fc_in(x)))
        for block in self.res_blocks:
            residual = block(h)
            h = h + residual if h.shape == residual.shape else residual
        return self.actor_mean(h), self.critic(h)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(x)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


NETWORK_REGISTRY = {
    "mlp-small":   lambda od, na: MLPActorCritic(od, na, hidden=[128, 128]),
    "mlp-medium":  lambda od, na: MLPActorCritic(od, na, hidden=[256, 256]),
    "mlp-large":   lambda od, na: MLPActorCritic(od, na, hidden=[512, 512]),
    "resmlp":      lambda od, na: ResMLPActorCritic(od, na, hidden=[256, 256]),
}
