# MuJoCo Environment Setup Guide

How to add a new MuJoCo Gymnasium environment to a training project — from discovery to deployment.

## 1. Discover Environment Specs

Never hardcode observation/action dimensions. Always read them from the live environment:

```python
import gymnasium as gym
import numpy as np

env_id = "HalfCheetah-v5"  # always use v5 (highest version)

env = gym.make(env_id)

obs_dim   = env.observation_space.shape[0]
act_dim   = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
max_steps  = env.spec.max_episode_steps
threshold  = env.spec.reward_threshold  # may be None

print(f"obs_dim={obs_dim} act_dim={act_dim} action_range=[{min_action}, {max_action}]")
print(f"max_steps={max_steps} threshold={threshold}")

env.close()
```

Key rule: action bounds vary across envs. Most are [-1,1], but InvertedPendulum is [-3,3], Humanoid is [-0.4,0.4], Pusher is [-2,2]. Never assume `max_action = 1.0`.

## 2. Choose Network Architecture

Match network size to environment complexity:

| Obs Dim | Act Dim | Hidden | Network Key | Example Envs |
|---------|---------|--------|-------------|-------------|
| ≤ 10 | 1–2 | [128, 128] | mlp-small | InvertedPendulum, Swimmer, Reacher |
| 11–23 | 3–7 | [256, 256] | mlp-medium | Hopper, HalfCheetah, Walker2d, Pusher |
| ≥ 100 | 8 | [512, 512] | mlp-large | Ant |
| ≥ 300 | 17 | [512, 512] or [512, 256] | mlp-large | Humanoid, HumanoidStandup |

### Continuous-action network template

```python
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    """Gaussian policy for continuous actions."""
    def __init__(self, obs_dim, act_dim, hidden=[256, 256], max_action=1.0):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.max_action = max_action
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean(h)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Squash through tanh and scale
        action = torch.tanh(action) * self.max_action
        return action, log_prob, dist.entropy().sum(dim=-1)
```

## 3. Wrap the Environment

Standard MuJoCo wrapper stack:

```python
import gymnasium as gym
import numpy as np

def make_env(env_id, seed=42):
    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)           # safety: clamp actions
    env = gym.wrappers.NormalizeObservation(env)  # running mean/std
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10))   # prevent stat explosion
    env.reset(seed=seed)
    return env
```

For vectorized training (PPO):

```python
def make_vec_env(env_id, num_envs=4, seed=42):
    def _make(rank):
        def _init():
            return make_env(env_id, seed + rank)
        return _init

    envs = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    return envs
```

## 4. Choose Algorithm and Hyperparameters

Decision matrix based on environment difficulty:

| Tier | Time to Solve | Algorithm | Key Hyperparams |
|------|--------------|-----------|----------------|
| Trivial (<1 min) | Any | Any | defaults work |
| Easy (1–2 min) | SAC, TD3 | lr=3e-4, buffer=1M, batch=256 |
| Medium (2–5 min) | SAC | lr=3e-4, alpha auto-tuned |
| Hard (5–10 min) | SAC | lr=3e-4, larger network, longer warmup |
| Extreme (>10 min) | SAC | Use Kaggle P100 (not Colab T4) |

### SAC defaults (best general-purpose choice)

```python
sac_config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "start_steps": 10_000,      # random exploration before learning
    "policy_freq": 2,            # actor updated every 2 critic updates
    "target_entropy": -act_dim,  # auto-tuned
    "init_temperature": 1.0,
}
```

### TD3 defaults (simpler alternative)

```python
td3_config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "policy_delay": 2,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "exploration_noise": 0.1,
}
```

### PPO defaults (if on-policy needed)

```python
ppo_config = {
    "lr": 3e-4,           # higher than SAC (1e-4 is too low)
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_epochs": 10,
    "n_minibatches": 32,
    "num_envs": 4,         # at least 4 for stable gradients
    "num_steps": 512,      # shorter rollouts, more frequent updates
}
```

## 5. Estimate Training Time

```
total_steps ÷ estimated_steps_per_sec ÷ 60 = minutes

T4 GPU step rates:
  Off-policy (SAC/TD3): ~2000 steps/sec  (0.5ms per env-step + update)
  On-policy  (PPO):     ~5000 steps/sec  (no replay buffer overhead)
  Humanoid-size models: ~600 steps/sec   (large network forward/backward)
```

Colab free-tier GPU dies after ~10 minutes. Plan training to complete within this window:

| Env | Steps Needed | SAC Time | PPO Time | Fits Colab? |
|-----|-------------|----------|----------|------------|
| InvertedPendulum | 50k | 25s | 10s | Yes |
| Swimmer | 100k | 50s | 20s | Yes |
| Hopper | 200k | 100s | 40s | Yes |
| HalfCheetah | 200k | 100s | 40s | Yes |
| Walker2d | 200k | 100s | 40s | Yes |
| Ant | 500k | 250s | 100s | Yes |
| Humanoid | 2M | 1000s | 400s | **No** → Kaggle |

## 6. Set Up Output Structure

Every training script must produce three artifacts for cron-watchtower monitoring:

```
output/{env_name}/{algo}/
├── train.log            # timestamped per-N-episodes
├── metrics.csv          # one row per episode, crash-safe append
├── training_curves.png  # 4-panel, overwritten every N episodes
└── best_model.pt        # weights-only checkpoint
```

### Log format

```
[HH:MM:SS] Ep 1200 | reward=203 | avg100=153.6 | q_mean=0.347 | alpha=0.12 | elapsed=17s
```

### CSV columns

```
episode,reward,steps,avg100_reward,eval_reward,actor_loss,critic_loss,q_mean,alpha,elapsed_s
```

### PNG panels (4-panel 2×2)

1. **Reward curve** — raw + avg100 moving average + solved threshold line
2. **Losses** — actor loss + critic loss (or PG loss + VF loss for PPO)
3. **Diagnostics** — Q-value mean or entropy or alpha temperature
4. **Eval performance** — eval reward vs training steps

## 7. Colab Deployment Checklist

Before pushing to Colab:

```bash
# 1. Lint
ruff check .

# 2. Verify env exists on target platform
python -c "import gymnasium; gym.make('ENV_NAME-v5')"

# 3. Estimate time budget
# total_steps ÷ 2000 ÷ 60 = minutes (SAC/TD3)
# total_steps ÷ 5000 ÷ 60 = minutes (PPO)

# 4. Check network input dim matches env
python -c "
import gymnasium as gym
env = gym.make('ENV_NAME-v5')
print(f'obs={env.observation_space.shape[0]} act={env.action_space.shape[0]}')
env.close()
"
```

### Launch pattern

```bash
# Provision
colab new --gpu T4 -s training

# Upload training script
colab upload train.py /content/train.py

# Launch detached (installs deps, spawns training)
colab exec -f launch.py --timeout 120

# Monitor via cron watchtower (every 2-5 min)
# See CLAUDE.md "Cron watchtower for long-running Colab jobs"
```

### Minimal launcher (`launch.py`)

```python
import subprocess, sys, os

DEPS = ["gymnasium[mujoco]", "matplotlib"]
for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

os.makedirs("/content/output", exist_ok=True)
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["MUJOCO_GL"] = "egl"

with open("/content/output/launch.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"OK. PID={proc.pid}")
```

## 8. Common Pitfalls

### Hardcoded action bounds
```python
# WRONG:
action = torch.tanh(net_output)  # assumes [-1, 1]

# RIGHT:
max_a = float(env.action_space.high[0])
action = torch.tanh(net_output) * max_a
```

### Wrong observation dimension
```python
# WRONG — using v4 dims with v5 env:
Ant-v5: obs_dim=27   # v5 is 105

# RIGHT — read from env at runtime:
obs_dim = env.observation_space.shape[0]
```

### Too few vectorized envs for PPO
```python
# WRONG:
num_envs = 1   # gradients too noisy, policy doesn't learn

# RIGHT:
num_envs = 4   # minimum for stable PPO
```

### Learning rate too low
```python
# WRONG — common mistake with PPO on MuJoCo:
lr = 1e-4   # too slow, policy stuck at random

# RIGHT:
lr = 3e-4   # CleanRL default, works on all envs
```

### Not clipping observations
```python
# WRONG — unbounded obs can explode NormalizeObservation stats:
env = gym.make(env_id)
env = gym.wrappers.NormalizeObservation(env)

# RIGHT:
env = gym.make(env_id)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
```

### Using Pusher-v4 with mujoco>=3
```python
# WRONG:
env = gym.make("Pusher-v4")   # error: only supported on mujoco<3

# RIGHT:
env = gym.make("Pusher-v5")
```

### Forgetting headless rendering
```python
# Set BEFORE importing gymnasium or in launch script:
os.environ["MUJOCO_GL"] = "egl"
```

## Quick Start: New Env in 5 Lines

```python
import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v5")
obs_dim = env.observation_space.shape[0]     # 17
act_dim = env.action_space.shape[0]          # 6
max_a   = float(env.action_space.high[0])    # 1.0

# Now plug obs_dim, act_dim, max_a into your network and algorithm.
```

That's the core. Everything else — network size, hyperparameters, timesteps — follows from the tables above.
