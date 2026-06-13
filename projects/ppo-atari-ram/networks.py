"""Actor-Critic networks for Atari RAM (128-dim vector input)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class MLPActorCritic(nn.Module):
    """Shared-trunk MLP: obs -> hidden -> [actor_logits, value]."""

    def __init__(self, obs_dim: int = 128, n_actions: int = 4,
                 hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
            ])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.actor = nn.Linear(in_dim, n_actions)
        self.critic = nn.Linear(in_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=1.0)
        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor(h), self.critic(h)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


class ResMLPActorCritic(nn.Module):
    """MLP with residual skip connections. Deeper but stable via skip."""

    def __init__(self, obs_dim: int = 128, n_actions: int = 4,
                 hidden: list = None):
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

        self.actor = nn.Linear(hidden[-1], n_actions)
        self.critic = nn.Linear(hidden[-1], 1)
        self.init_weights()

    def init_weights(self):
        for m in [self.fc_in] + list(self.res_blocks):
            for sub in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(sub, nn.Linear):
                    orthogonal_init(sub, gain=1.0)
        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.ln_in(self.fc_in(x)))
        for block in self.res_blocks:
            h = h + block(h) if h.shape == block(h).shape else block(h)
        return self.actor(h), self.critic(h)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


NETWORK_REGISTRY = {
    "mlp-small":   lambda na: MLPActorCritic(n_actions=na, hidden=[128, 128]),
    "mlp-medium":  lambda na: MLPActorCritic(n_actions=na, hidden=[256, 256]),
    "mlp-large":   lambda na: MLPActorCritic(n_actions=na, hidden=[512, 512]),
    "resmlp":      lambda na: ResMLPActorCritic(n_actions=na, hidden=[256, 256]),
}
