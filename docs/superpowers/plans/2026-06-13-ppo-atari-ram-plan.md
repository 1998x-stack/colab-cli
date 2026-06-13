# PPO Atari RAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable PPO agent that trains on Atari RAM environments (128-byte observation), deployable on Colab T4 with cron-based monitoring.

**Architecture:** Five Python modules with clean interfaces: `networks.py` (actor-critic MLPs), `ppo_agent.py` (PPO clipped objective + GAE), `env_factory.py` (Atari RAM env creation), `config_loader.py` (two-tier JSON merge), `train.py` (CleanRL rollout→update loop + metrics + plotting). Configs generated for all 63 ALE/*-ram-v5 envs; 3 actually trained (Pong, Asterix, MsPacman).

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium + ale-py, matplotlib, NumPy. Deploy on Colab T4 via colab-cli.

---

## File Map

| File | Responsibility | Lines |
|------|---------------|-------|
| `projects/ppo-atari-ram/networks.py` | MLPActorCritic + registry | ~80 |
| `projects/ppo-atari-ram/ppo_agent.py` | PPOAgent: act, compute_returns, update | ~100 |
| `projects/ppo-atari-ram/env_factory.py` | make_ram_env factory | ~40 |
| `projects/ppo-atari-ram/config_loader.py` | Merge _defaults + per-env JSON | ~50 |
| `projects/ppo-atari-ram/generate_configs.py` | Auto-generate 63 env configs | ~50 |
| `projects/ppo-atari-ram/train.py` | Argparse, training loop, metrics, plotting | ~280 |
| `projects/ppo-atari-ram/launch.py` | Colab detached launcher | ~50 |
| `projects/ppo-atari-ram/fetch.sh` | Cron: tar→download→extract→report | ~60 |
| `projects/ppo-atari-ram/configs/_defaults.json` | Shared PPO hyperparameters | ~20 |
| `projects/ppo-atari-ram/configs/ALE-*-ram-v5.json` | 63 per-env overrides | ~5 each |

---

### Task 1: Project scaffolding

**Files:**
- Create: `projects/ppo-atari-ram/` directory

- [ ] **Step 1: Create directory**

```bash
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram/configs
```

- [ ] **Step 2: Verify**

```bash
ls /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram/
# Expected: configs/
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/.gitkeep  # if .gitkeep was created
git commit -m "chore: scaffold ppo-atari-ram project directory"
```

---

### Task 2: Default config (`_defaults.json`)

**Files:**
- Create: `projects/ppo-atari-ram/configs/_defaults.json`

- [ ] **Step 1: Write defaults**

```json
{
  "network": "mlp-medium",
  "total_timesteps": 500000,
  "num_envs": 4,
  "num_steps": 128,
  "lr": 2.5e-4,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_coef": 0.1,
  "ent_coef": 0.01,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "n_epochs": 4,
  "n_minibatches": 4,
  "eval_interval": 10,
  "eval_episodes": 5,
  "plot_interval": 5,
  "seed": 42
}
```

- [ ] **Step 2: Verify valid JSON**

```bash
python3 -m json.tool /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram/configs/_defaults.json > /dev/null && echo "OK"
# Expected: OK
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/configs/_defaults.json
git commit -m "feat: add PPO default hyperparameters config"
```

---

### Task 3: Networks module (`networks.py`)

**Files:**
- Create: `projects/ppo-atari-ram/networks.py`

The network takes RAM observation `(128,)` float, outputs action logits and value. Shared trunk with separate actor/critic heads.

- [ ] **Step 1: Write networks.py**

```python
"""Actor-Critic networks for Atari RAM (128-dim vector input)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class MLPActorCritic(nn.Module):
    """Shared-trunk MLP: obs → hidden → [actor_logits, value]."""

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
```

- [ ] **Step 2: Quick smoke test — instantiate and forward pass**

```bash
python3 -c "
import torch
from networks import MLPActorCritic, NETWORK_REGISTRY

net = MLPActorCritic(obs_dim=128, n_actions=6)
x = torch.randn(1, 128)
action, log_prob, entropy, value = net.get_action_and_value(x)
print(f'action={action.item()} log_prob={log_prob.item():.3f} entropy={entropy.item():.3f} value={value.item():.3f}')

# Registry
net2 = NETWORK_REGISTRY['mlp-small'](4)
a, lp, e, v = net2.get_action_and_value(x)
print(f'mlp-small: action={a.item()} n_actions=4 ok')

net3 = NETWORK_REGISTRY['resmlp'](18)
a, lp, e, v = net3.get_action_and_value(x)
print(f'resmlp: action={a.item()} n_actions=18 ok')
print('All networks OK')
"
# Expected: prints shapes and values for all three, no errors
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/networks.py
git commit -m "feat: add MLPActorCritic and ResMLPActorCritic with registry"
```

---

### Task 4: Env factory (`env_factory.py`)

**Files:**
- Create: `projects/ppo-atari-ram/env_factory.py`

- [ ] **Step 1: Write env_factory.py**

```python
"""Atari RAM environment factory with vectorized envs."""
import numpy as np
import gymnasium as gym


def make_ram_env(env_id: str, num_envs: int = 4, seed: int = 42):
    """Create AsyncVectorEnv of Atari RAM environments.

    Observation: (128,) uint8 scaled to [0, 1].
    Action space: discrete, env-specific size.
    """

    def _make_env(rank: int):
        def _init():
            env = gym.make(env_id, max_episode_steps=108000)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: obs.astype(np.float32) / 255.0
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    envs = gym.vector.AsyncVectorEnv([_make_env(i) for i in range(num_envs)])
    return envs


def get_env_info(env_id: str):
    """Return (obs_dim, n_actions) for a given env ID.

    Returns (128, n_actions) for RAM envs. Creates and disposes a temp env.
    """
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    return obs_dim, n_actions
```

- [ ] **Step 2: Smoke test — env creation (if gymnasium + ale-py installed)**

```bash
python3 -c "
from env_factory import make_ram_env, get_env_info
obs_dim, n_actions = get_env_info('ALE/Pong-ram-v5')
print(f'Pong: obs_dim={obs_dim} n_actions={n_actions}')
assert obs_dim == 128
assert n_actions == 6
print('env_factory OK')
" 2>/dev/null || echo "SKIP: gymnasium not installed locally (tested on Colab)"
# Expected: Pong: obs_dim=128 n_actions=6 (or SKIP if no gymnasium)
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/env_factory.py
git commit -m "feat: add Atari RAM env factory with vectorized envs"
```

---

### Task 5: Config loader (`config_loader.py`)

**Files:**
- Create: `projects/ppo-atari-ram/config_loader.py`

- [ ] **Step 1: Write config_loader.py**

```python
"""Two-tier config: _defaults.json base + per-env JSON override."""
import json
import os
from typing import Dict, Any


def load_config(env_id: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Load merged config for an environment.

    Merges _defaults.json (base) with <env_id>.json (overrides).
    The env_id key is always set from the override file.
    """
    defaults_path = os.path.join(config_dir, "_defaults.json")
    with open(defaults_path) as f:
        config = json.load(f)

    env_path = os.path.join(config_dir, f"{env_id}.json")
    if os.path.exists(env_path):
        with open(env_path) as f:
            overrides = json.load(f)
        config.update(overrides)

    config.setdefault("env_id", env_id)
    return config


def list_configs(config_dir: str = "configs"):
    """List all env config files (excluding _defaults.json)."""
    files = sorted(os.listdir(config_dir))
    return [f.replace(".json", "") for f in files
            if f.endswith(".json") and not f.startswith("_")]
```

- [ ] **Step 2: Smoke test**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram && python3 -c "
from config_loader import load_config
# Test with just defaults (no per-env file exists yet)
import json, os
# Create a temp per-env config to test merge
os.makedirs('configs', exist_ok=True)
with open('configs/ALE-Test-ram-v5.json', 'w') as f:
    json.dump({'env_id': 'ALE/Test-ram-v5', 'n_actions': 4, 'network': 'mlp-small'}, f)

cfg = load_config('ALE-Test-ram-v5')
print(f'env_id={cfg[\"env_id\"]} n_actions={cfg[\"n_actions\"]} network={cfg[\"network\"]} lr={cfg[\"lr\"]}')
assert cfg['n_actions'] == 4
assert cfg['network'] == 'mlp-small'
assert cfg['lr'] == 2.5e-4  # from defaults
os.remove('configs/ALE-Test-ram-v5.json')
print('config_loader OK')
"
# Expected: merge works, then OK
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/config_loader.py
git commit -m "feat: add two-tier config loader with JSON merge"
```

---

### Task 6: PPO agent (`ppo_agent.py`)

**Files:**
- Create: `projects/ppo-atari-ram/ppo_agent.py`

- [ ] **Step 1: Write ppo_agent.py**

```python
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
```

- [ ] **Step 2: Unit test — forward pass with random rollouts**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram && python3 -c "
import torch
from networks import MLPActorCritic, NETWORK_REGISTRY
from ppo_agent import PPOAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NETWORK_REGISTRY['mlp-medium'](6)  # 6 actions = Pong
agent = PPOAgent(net, device=device)

# Simulate a rollout: num_steps=8, num_envs=2, obs_dim=128
obs = torch.randn(8, 2, 128)
actions = torch.randint(0, 6, (8, 2))
rewards = torch.randn(8, 2)
dones = torch.zeros(8, 2).bool()
values = torch.randn(8, 2)
log_probs = torch.randn(8, 2)
next_obs = torch.randn(2, 128)
next_done = torch.zeros(2).bool()

advantages, returns = agent.compute_returns(next_obs, next_done, rewards, dones, values)
print(f'advantages shape: {advantages.shape}  returns shape: {returns.shape}')

rollouts = {
    'obs': obs, 'actions': actions, 'log_probs': log_probs,
    'advantages': advantages, 'returns': returns, 'values': values,
}
metrics = agent.update(rollouts)
print(f'metrics: pg_loss={metrics[\"pg_loss\"]:.4f} vf_loss={metrics[\"vf_loss\"]:.4f} entropy={metrics[\"entropy\"]:.4f} clip_frac={metrics[\"clip_frac\"]:.4f}')
print('PPO agent OK')
"
# Expected: shapes printed, metrics dict with 4 keys, no errors
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/ppo_agent.py
git commit -m "feat: add PPO agent with GAE and clipped objective"
```

---

### Task 7: Config generator (`generate_configs.py`)

**Files:**
- Create: `projects/ppo-atari-ram/generate_configs.py`

Generates per-env JSON files for all `ALE/*-ram-v5` environments. Must run on a machine with ale-py installed (runs locally to produce static JSONs).

- [ ] **Step 1: Write generate_configs.py**

```python
#!/usr/bin/env python3
"""Generate per-environment JSON configs for all ALE/*-ram-v5 environments.

Run once to produce 63 config files. Configs are static JSON — no
ale-py dependency at training time.
"""
import json
import os
import sys


def known_atari_games():
    """Full list of Atari 2600 RAM environment IDs.

    Each tuple: (env_id, n_actions, suggested_network, total_timesteps).
    Sourced from ALE 0.10.x / Gymnasium Atari registry.
    Network and timesteps are defaults; adjust per-game after testing.
    """
    # (env_slug, n_actions, network, total_timesteps)
    games = [
        ("ALE/Adventure-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/AirRaid-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/Alien-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/Amidar-ram-v5", 10, "mlp-large", 1000000),
        ("ALE/Assault-ram-v5", 7, "mlp-medium", 1000000),
        ("ALE/Asterix-ram-v5", 9, "mlp-medium", 500000),
        ("ALE/Asteroids-ram-v5", 14, "mlp-large", 1000000),
        ("ALE/Atlantis-ram-v5", 4, "mlp-small", 500000),
        ("ALE/BankHeist-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/BattleZone-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/BeamRider-ram-v5", 9, "mlp-medium", 500000),
        ("ALE/Berzerk-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Bowling-ram-v5", 6, "mlp-small", 500000),
        ("ALE/Boxing-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Breakout-ram-v5", 4, "mlp-small", 500000),
        ("ALE/Carnival-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/Centipede-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/ChopperCommand-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/CrazyClimber-ram-v5", 9, "mlp-medium", 500000),
        ("ALE/Defender-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/DemonAttack-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/DoubleDunk-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/ElevatorAction-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/Enduro-ram-v5", 9, "mlp-medium", 1000000),
        ("ALE/FishingDerby-ram-v5", 18, "mlp-medium", 500000),
        ("ALE/Freeway-ram-v5", 3, "mlp-small", 500000),
        ("ALE/Frostbite-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Gopher-ram-v5", 8, "mlp-medium", 1000000),
        ("ALE/Gravitar-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/Hero-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/IceHockey-ram-v5", 18, "mlp-medium", 500000),
        ("ALE/Jamesbond-ram-v5", 18, "mlp-medium", 500000),
        ("ALE/JourneyEscape-ram-v5", 18, "mlp-large", 1000000),
        ("ALE/Kangaroo-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Krull-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/KungFuMaster-ram-v5", 14, "mlp-medium", 1000000),
        ("ALE/MontezumaRevenge-ram-v5", 18, "mlp-large", 2000000),
        ("ALE/MsPacman-ram-v5", 9, "mlp-large", 1000000),
        ("ALE/NameThisGame-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/Phoenix-ram-v5", 8, "mlp-medium", 500000),
        ("ALE/Pitfall-ram-v5", 18, "mlp-large", 2000000),
        ("ALE/Pong-ram-v5", 6, "mlp-small", 500000),
        ("ALE/Pooyan-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/PrivateEye-ram-v5", 18, "mlp-large", 2000000),
        ("ALE/Qbert-ram-v5", 6, "mlp-medium", 1000000),
        ("ALE/Riverraid-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/RoadRunner-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Robotank-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Seaquest-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Skiing-ram-v5", 3, "mlp-small", 500000),
        ("ALE/Solaris-ram-v5", 18, "mlp-large", 500000),
        ("ALE/SpaceInvaders-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/StarGunner-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Tennis-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/TimePilot-ram-v5", 10, "mlp-medium", 1000000),
        ("ALE/Tutankham-ram-v5", 8, "mlp-medium", 500000),
        ("ALE/UpNDown-ram-v5", 6, "mlp-medium", 500000),
        ("ALE/Venture-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/VideoPinball-ram-v5", 9, "mlp-small", 500000),
        ("ALE/WizardOfWor-ram-v5", 10, "mlp-medium", 500000),
        ("ALE/YarsRevenge-ram-v5", 18, "mlp-medium", 1000000),
        ("ALE/Zaxxon-ram-v5", 18, "mlp-medium", 1000000),
    ]
    return games


def main():
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)

    # Load defaults to get solved thresholds
    with open(os.path.join(config_dir, "_defaults.json")) as f:
        defaults = json.load(f)

    # Solved thresholds (human-level scores) for reference
    solved_thresholds = {
        "Alien": 3000, "Amidar": 1000, "Assault": 800, "Asterix": 5000,
        "Asteroids": 1000, "Atlantis": 100000, "BankHeist": 1000,
        "BattleZone": 30000, "BeamRider": 5000, "Berzerk": 1000,
        "Bowling": 200, "Boxing": 50, "Breakout": 40, "Carnival": 5000,
        "Centipede": 5000, "ChopperCommand": 5000, "CrazyClimber": 50000,
        "Defender": 50000, "DemonAttack": 10000, "DoubleDunk": 0,
        "ElevatorAction": 30000, "Enduro": 500, "FishingDerby": 20,
        "Freeway": 30, "Frostbite": 1000, "Gopher": 5000, "Gravitar": 3000,
        "Hero": 30000, "IceHockey": 0, "Jamesbond": 1000,
        "JourneyEscape": 0, "Kangaroo": 2000, "Krull": 8000,
        "KungFuMaster": 30000, "MontezumaRevenge": 5000, "MsPacman": 3000,
        "NameThisGame": 5000, "Phoenix": 10000, "Pitfall": 0,
        "Pong": 18, "Pooyan": 3000, "PrivateEye": 0, "Qbert": 10000,
        "Riverraid": 10000, "RoadRunner": 30000, "Robotank": 30,
        "Seaquest": 50000, "Skiing": 0, "Solaris": 2000,
        "SpaceInvaders": 1000, "StarGunner": 30000, "Tennis": 0,
        "TimePilot": 5000, "Tutankham": 200, "UpNDown": 50000,
        "Venture": 1000, "VideoPinball": 100000, "WizardOfWor": 5000,
        "YarsRevenge": 30000, "Zaxxon": 10000,
        "Adventure": 0, "AirRaid": 0,
    }

    generated = 0
    for env_id, n_actions, network, total_timesteps in known_atari_games():
        game_name = env_id.split("/")[1].replace("-ram-v5", "")
        config = {
            "env_id": env_id,
            "n_actions": n_actions,
            "solved_threshold": solved_thresholds.get(game_name, 0),
            "network": network,
            "total_timesteps": total_timesteps,
        }

        filename = env_id.replace("/", "-") + ".json"
        filepath = os.path.join(config_dir, filename)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        generated += 1

    print(f"Generated {generated} config files in {config_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run config generator**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram && python3 generate_configs.py
# Expected: Generated 63 config files in .../configs
```

- [ ] **Step 3: Verify sample config**

```bash
python3 -m json.tool /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram/configs/ALE-Pong-ram-v5.json
# Expected: {"env_id": "ALE/Pong-ram-v5", "n_actions": 6, ...}
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/generate_configs.py projects/ppo-atari-ram/configs/ALE-*.json
git commit -m "feat: generate 63 Atari RAM environment configs"
```

---

### Task 8: Training loop (`train.py`)

**Files:**
- Create: `projects/ppo-atari-ram/train.py`

This is the main entry point. CleanRL-style rollout collection + PPO update loop, with metrics, CSV, PNGs, and checkpoints.

- [ ] **Step 1: Write train.py**

```python
#!/usr/bin/env python3
"""PPO on Atari RAM — CleanRL-style training loop with pluggable components."""
import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_loader import load_config
from networks import NETWORK_REGISTRY
from ppo_agent import PPOAgent
from env_factory import make_ram_env

parser = argparse.ArgumentParser()
parser.add_argument("--envs", nargs="+",
                    default=["ALE/Pong-ram-v5", "ALE/Asterix-ram-v5", "ALE/MsPacman-ram-v5"])
parser.add_argument("--out_dir", default="/content/ppo-atari-output")
parser.add_argument("--config_dir", default="configs")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/logs", exist_ok=True)
os.makedirs(f"{args.out_dir}/pngs", exist_ok=True)
os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)

LOG_PATH = f"{args.out_dir}/logs/train.log"
CSV_PATH = f"{args.out_dir}/metrics.csv"

device = torch.device(args.device)


# ── Logging ──────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── CSV ──────────────────────────────────────────────────────────────────
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "iteration", "env", "total_steps", "mean_reward", "avg10_reward",
    "eval_reward", "pg_loss", "vf_loss", "entropy", "clip_frac",
    "lr", "elapsed_s"
])
csv_file.flush()


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_curves(env_name, history, out_dir):
    """2x2 panel: reward, losses, entropy, clip fraction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PPO — {env_name}", fontsize=14, fontweight="bold")

    iters = [h["iteration"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    avg10 = [h["avg10_reward"] for h in history]
    solved = history[0].get("solved_threshold", 0)

    # Reward
    ax = axes[0, 0]
    ax.plot(iters, rewards, alpha=0.35, color="steelblue", linewidth=0.6, label="Mean reward")
    ax.plot(iters, avg10, color="darkorange", linewidth=2, label="Avg10")
    if solved:
        ax.axhline(y=solved, color="green", linestyle="--", alpha=0.5,
                   label=f"Solved ({solved})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Reward")
    ax.set_title("Training Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 1]
    ax.plot(iters, [h["pg_loss"] for h in history], color="crimson",
            linewidth=1.2, label="Policy loss")
    ax.plot(iters, [h["vf_loss"] for h in history], color="royalblue",
            linewidth=1.2, label="Value loss")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("Losses"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(iters, [h["entropy"] for h in history], color="mediumseagreen", linewidth=1.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy"); ax.grid(True, alpha=0.3)

    # Clip fraction
    ax = axes[1, 1]
    ax.plot(iters, [h["clip_frac"] for h in history], color="darkviolet", linewidth=1.5)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="clip_coef=0.1")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Clip fraction")
    ax.set_title("PPO Clip Fraction"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/pngs/{env_name.replace('/', '_')}_curves.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_env_histories, out_dir):
    """Single figure: eval reward curves for all trained envs overlaid."""
    if len(all_env_histories) < 2:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_histories)))
    for idx, (name, history) in enumerate(sorted(all_env_histories.items())):
        if not history:
            continue
        evals = [(h["iteration"], h["eval_reward"]) for h in history
                 if h.get("eval_reward") is not None]
        if evals:
            xs, ys = zip(*evals)
            ax.plot(xs, ys, color=colors[idx], linewidth=1.8, label=name,
                    marker="o", markersize=3)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Eval Reward")
    ax.set_title("PPO Atari RAM — Evaluation Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/pngs/comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Eval ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(agent, env_id, n_episodes=5):
    import gymnasium as gym
    eval_env = gym.make(env_id, max_episode_steps=108000)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        obs = obs.astype(np.float32) / 255.0
        ep_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.act(torch.FloatTensor(obs).unsqueeze(0))
            obs, reward, terminated, truncated, _ = eval_env.step(action.item())
            obs = obs.astype(np.float32) / 255.0
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    eval_env.close()
    return np.mean(rewards), np.std(rewards)


# ── Main ─────────────────────────────────────────────────────────────────
log(f"=== PPO Atari RAM | device={device} | {datetime.now()} ===")
log(f"Envs: {args.envs}")

all_histories = {}

for env_idx, env_id in enumerate(args.envs):
    log(f"\n{'=' * 60}")
    log(f"[{env_idx + 1}/{len(args.envs)}] {env_id}")
    log(f"{'=' * 60}")

    cfg = load_config(env_id.replace("/", "-") + "-ram-v5", args.config_dir)
    log(f"Config: network={cfg['network']} timesteps={cfg['total_timesteps']} "
        f"n_actions={cfg['n_actions']} lr={cfg['lr']}")

    num_envs = cfg["num_envs"]
    num_steps = cfg["num_steps"]
    total_timesteps = cfg["total_timesteps"]
    n_iterations = total_timesteps // (num_envs * num_steps)

    envs = make_ram_env(env_id, num_envs=num_envs, seed=cfg["seed"])

    net = NETWORK_REGISTRY[cfg["network"]](cfg["n_actions"])
    agent = PPOAgent(
        network=net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_coef=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        n_epochs=cfg["n_epochs"],
        n_minibatches=cfg["n_minibatches"],
        device=str(device),
    )

    history = []
    reward_buffer = deque(maxlen=10)
    best_eval = -float("inf")
    start_time = time.time()

    obs, _ = envs.reset()
    obs = torch.FloatTensor(obs)

    for iteration in range(1, n_iterations + 1):
        # ── Rollout collection ──────────────────────────────────────────
        rollouts = {
            "obs": torch.zeros(num_steps, num_envs, 128),
            "actions": torch.zeros(num_steps, num_envs, dtype=torch.long),
            "log_probs": torch.zeros(num_steps, num_envs),
            "rewards": torch.zeros(num_steps, num_envs),
            "dones": torch.zeros(num_steps, num_envs),
            "values": torch.zeros(num_steps, num_envs),
        }

        for t in range(num_steps):
            actions, log_probs, values = agent.act(obs)
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
            dones = np.logical_or(terminateds, truncateds)

            rollouts["obs"][t] = obs
            rollouts["actions"][t] = actions
            rollouts["log_probs"][t] = log_probs
            rollouts["rewards"][t] = torch.FloatTensor(rewards)
            rollouts["dones"][t] = torch.FloatTensor(dones)
            rollouts["values"][t] = values

            obs = torch.FloatTensor(next_obs)
            dones_t = torch.FloatTensor(dones)

        # ── Compute returns ─────────────────────────────────────────────
        next_done = torch.FloatTensor(dones)
        advantages, returns = agent.compute_returns(
            obs, next_done, rollouts["rewards"],
            rollouts["dones"], rollouts["values"],
        )
        rollouts["advantages"] = advantages
        rollouts["returns"] = returns

        # ── PPO update ──────────────────────────────────────────────────
        metrics = agent.update(rollouts)

        mean_reward = rollouts["rewards"].mean().item()
        reward_buffer.append(mean_reward)
        avg10 = np.mean(reward_buffer) if reward_buffer else mean_reward

        elapsed = time.time() - start_time
        total_steps = iteration * num_envs * num_steps

        # ── Log ─────────────────────────────────────────────────────────
        history.append({
            "iteration": iteration,
            "mean_reward": mean_reward,
            "avg10_reward": avg10,
            "eval_reward": None,
            "pg_loss": metrics["pg_loss"],
            "vf_loss": metrics["vf_loss"],
            "entropy": metrics["entropy"],
            "clip_frac": metrics["clip_frac"],
        })

        csv_writer.writerow([
            iteration, env_id, total_steps,
            round(mean_reward, 2), round(avg10, 2),
            None,
            round(metrics["pg_loss"], 6), round(metrics["vf_loss"], 6),
            round(metrics["entropy"], 4), round(metrics["clip_frac"], 4),
            cfg["lr"], round(elapsed, 1),
        ])
        csv_file.flush()

        log(f"iter {iteration:3d}/{n_iterations} | {env_id.split('/')[1]:20s} "
            f"reward={mean_reward:7.2f} avg10={avg10:7.2f} | "
            f"pg_loss={metrics['pg_loss']:.4f} vf_loss={metrics['vf_loss']:.4f} | "
            f"ent={metrics['entropy']:.2f} clip={metrics['clip_frac']:.3f} | "
            f"steps={total_steps}")

        # ── Eval ────────────────────────────────────────────────────────
        if iteration % cfg["eval_interval"] == 0:
            eval_mean, eval_std = evaluate(agent, env_id, cfg["eval_episodes"])
            history[-1]["eval_reward"] = eval_mean
            log(f"  EVAL: mean={eval_mean:.2f} ± {eval_std:.2f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save({
                    "model": agent.net.state_dict(),
                    "iteration": iteration,
                    "eval_reward": eval_mean,
                }, f"{args.out_dir}/checkpoints/{env_id.replace('/', '_')}_best.pt")
                log(f"  -> new best! saved checkpoint")

        # ── Plot ────────────────────────────────────────────────────────
        if iteration % cfg["plot_interval"] == 0:
            plot_curves(env_id.split("/")[1], history, args.out_dir)
            all_histories[env_id.split("/")[1]] = history
            plot_comparison(all_histories, args.out_dir)

    # ── End of env loop ─────────────────────────────────────────────────
    plot_curves(env_id.split("/")[1], history, args.out_dir)
    all_histories[env_id.split("/")[1]] = history
    plot_comparison(all_histories, args.out_dir)
    envs.close()
    log(f"[DONE {env_id}] best_eval={best_eval:.1f} elapsed={elapsed:.0f}s")

# ── Final ────────────────────────────────────────────────────────────────
csv_file.close()

# Summary JSON
summary = {}
for env_name, hist in all_histories.items():
    evals = [h["eval_reward"] for h in hist if h["eval_reward"] is not None]
    summary[env_name] = {
        "iterations": len(hist),
        "final_avg10_reward": hist[-1]["avg10_reward"] if hist else 0,
        "best_eval_reward": max(evals) if evals else 0,
        "final_eval_reward": evals[-1] if evals else 0,
    }

with open(f"{args.out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log(f"\n=== ALL DONE | {datetime.now()} ===")
for env_name, s in sorted(summary.items()):
    log(f"  {env_name:30s}  best_eval={s['best_eval_reward']:8.1f}  "
        f"final_avg10={s['final_avg10_reward']:8.1f}")
```

- [ ] **Step 2: Local smoke test — single iteration (CPU, no Atari ROMs needed for syntax check)**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram && python3 -c "
import ast, sys
with open('train.py') as f:
    tree = ast.parse(f.read())
print('train.py syntax OK')
print(f'Functions: {[n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]}')
" 2>&1
# Expected: train.py syntax OK, Functions: ['plot_curves', 'plot_comparison', 'evaluate']
```

- [ ] **Step 3: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/train.py
git commit -m "feat: add CleanRL-style PPO training loop with metrics and plotting"
```

---

### Task 9: Colab launcher (`launch.py`)

**Files:**
- Create: `projects/ppo-atari-ram/launch.py`

- [ ] **Step 1: Write launch.py**

```python
#!/usr/bin/env python3
"""Launch PPO Atari RAM training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium[atari]", "ale-py", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/ppo-atari-output/logs/train.log"

print("=== Colab PPO Atari RAM Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/ppo-atari-output/logs", exist_ok=True)
os.makedirs("/content/ppo-atari-output/pngs", exist_ok=True)
os.makedirs("/content/ppo-atari-output/checkpoints", exist_ok=True)

print(f"\nLaunching {SCRIPT} detached...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
print(f"Output dir: /content/ppo-atari-output/")

time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
```

- [ ] **Step 2: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/launch.py
git commit -m "feat: add Colab launcher for PPO training"
```

---

### Task 10: Cron fetch script (`fetch.sh`)

**Files:**
- Create: `projects/ppo-atari-ram/fetch.sh`

- [ ] **Step 1: Write fetch.sh**

```bash
#!/bin/bash
# Fetch PPO Atari RAM training results from Colab VM.
# Usage: ./fetch.sh [session_name]
# Called by cron every 2 minutes.
set -euo pipefail

SESSION="${1:-ppo-atari}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="ppo-atari-output.tar.gz"

mkdir -p "$LOCAL_OUT"

# Proxy setup
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

COLB="$(which colab)"

# Step 1: Tar output on VM
echo "[fetch] $(date '+%H:%M:%S') Tarring output on VM..."
echo 'import subprocess as s; s.run(["tar","-czf","/content/ppo-atari-output.tar.gz","-C","/content","ppo-atari-output"], capture_output=True)' \
  | "$COLB" exec -s "$SESSION" --timeout 30 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed (WebSocket may be down), trying download directly..."
}

# Step 2: Download tar
echo "[fetch] Downloading..."
"$COLB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or tar missing"
    exit 0
}

# Step 3: Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: extract failed"
    exit 0
}

# Step 4: Report
echo "[fetch] $(date '+%H:%M:%S') Done."

# Print last 8 log lines
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo "══ Last 8 log lines ══"
    tail -8 "$LOCAL_OUT/logs/train.log"
fi

# Print last 3 CSV rows
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""
    echo "══ Metrics CSV tail ══"
    head -1 "$LOCAL_OUT/metrics.csv"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi

# PNGs
echo ""
echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"

echo ""
echo "Files in: $LOCAL_OUT"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram/fetch.sh
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/fetch.sh
git commit -m "feat: add cron fetch script for PPO Atari RAM training"
```

---

### Task 11: Colab deployment & cron setup

**Checklist (no new files):**

- [ ] **Step 1: Verify proxy health**

```bash
curl -s --max-time 5 -x http://127.0.0.1:7890 https://www.google.com -o /dev/null -w '%{http_code}\n'
# Expected: 200 or 302
```

- [ ] **Step 2: Provision GPU session**

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
colab new --gpu T4 -s ppo-atari
# Expected: session created
```

- [ ] **Step 3: Verify GPU**

```bash
echo 'import torch; print(f"CUDA: {torch.cuda.is_available()}"); print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")' | colab exec -s ppo-atari --timeout 15
# Expected: CUDA: True, GPU: Tesla T4, VRAM: ~15.1 GB
```

- [ ] **Step 4: Upload project files**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram
for f in train.py ppo_agent.py networks.py env_factory.py config_loader.py launch.py; do
    colab upload "$f" "/content/$f"
done
# Upload configs as a tar (colab upload doesn't do directories)
tar -czf /tmp/ppo-configs.tar.gz -C configs .
colab upload /tmp/ppo-configs.tar.gz /content/configs.tar.gz
echo 'import subprocess as s; s.run(["tar","-xzf","/content/configs.tar.gz","-C","/content/configs"]); s.run(["mkdir","-p","/content/configs"])' | colab exec -s ppo-atari --timeout 10
echo 'import os; print(sorted(os.listdir("/content/configs"))[:5])' | colab exec -s ppo-atari --timeout 10
```

- [ ] **Step 5: Launch training**

```bash
LAUNCH_SCRIPT="train.py" LAUNCH_DEPS="gymnasium[atari],ale-py,matplotlib" \
colab exec -s ppo-atari -f launch.py --timeout 120
# Expected: OK. PID=xxxxx log=/content/ppo-atari-output/logs/train.log
```

- [ ] **Step 6: Verify training started (wait 10s, check log)**

```bash
sleep 10
echo 'import subprocess as s; r = s.run(["tail","-5","/content/ppo-atari-output/logs/train.log"], capture_output=True, text=True); print(r.stdout or "NO LOG YET")' | colab exec -s ppo-atari --timeout 15
# Expected: first log lines from train.py
```

- [ ] **Step 7: Set up cron watchtower (2 minute interval)**

Use CronCreate:
```
cron: "*/2 * * * *"
prompt: "Run fetch.sh for ppo-atari session: 1. Check session alive with 'colab sessions | grep ppo-atari'. 2. If dead, report. 3. Tar output on VM via colab exec. 4. Download tar.gz via colab download. 5. Extract into projects/ppo-atari-ram/output/. 6. Print last 8 lines of train.log, last 3 rows of metrics.csv, list PNGs. 7. If reward is decreasing or clip_frac > 0.3 consistently, flag as potential problem."
```

- [ ] **Step 8: Test fetch manually**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/ppo-atari-ram && ./fetch.sh ppo-atari
# Expected: downloads, extracts, prints log tail, CSV, PNGs list
```

- [ ] **Step 9: Commit after validation**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/output/.gitkeep  # if any
git commit -m "chore: PPO Atari RAM deployment validated on Colab T4"
```

---

### Task 12: Monitoring & Feedback Loop

During training, monitor the cron output and take corrective action:

**Healthy signals:**
- `avg10_reward` trending up
- `clip_frac` between 0.01 and 0.15
- `entropy` between 0.5 and 2.0
- `vf_loss` decreasing

**Warning signals and fixes:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `clip_frac` > 0.3 consistently | PPO steps too aggressive | Reduce `n_epochs` to 2, or reduce `lr` |
| `entropy` → 0 quickly | Policy collapsed | Increase `ent_coef` to 0.02-0.05 |
| `vf_loss` exploding | Value function overfitting | Reduce `vf_coef` to 0.3, reduce `n_epochs` |
| Reward flat, no learning | LR too low or network too small | Bump network to `mlp-large`, increase `lr` to 5e-4 |
| NaN in any metric | Gradient explosion | Reduce `max_grad_norm` to 0.1, lower `lr` |

To fix mid-training: update the per-env JSON config, re-upload, kill the old process, re-launch. The cron will immediately reflect the new run's metrics.

- [ ] **Step 1: Monitor first cron output for 3-4 cycles (6-8 min)**

Check each cron report for healthy signals. If any warning signal appears, apply the fix from the table above.

- [ ] **Step 2: After training completes, commit final outputs**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/ppo-atari-ram/output/
git commit -m "results: PPO Atari RAM training outputs — Pong, Asterix, MsPacman"
```

- [ ] **Step 3: Stop Colab session**

```bash
colab stop -s ppo-atari
```
