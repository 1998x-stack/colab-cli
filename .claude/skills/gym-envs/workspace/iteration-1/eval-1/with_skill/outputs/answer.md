# HalfCheetah-v5 for TD3 Training

## Obs/Action Dimensions

| Property | Value |
|----------|-------|
| Observation dim | **17** (Box, unbounded) |
| Action dim | **6** (Box, continuous [-1, 1]) |
| Max action | **1.0** |

Always read from live env:
```python
obs_dim = env.observation_space.shape[0]   # 17
act_dim = env.action_space.shape[0]        # 6
max_action = float(env.action_space.high[0])  # 1.0
```

## Action Range

**For HalfCheetah-v5, YES — action range is [-1, 1].**

But NOT universal. Several MuJoCo envs have different bounds:

| Env | Action Bounds |
|-----|--------------|
| HalfCheetah, Hopper, Walker2d, Ant, Swimmer, Reacher | [-1, 1] |
| InvertedPendulum-v5 | **[-3, 3]** |
| Pusher-v5 | **[-2, 2]** |
| Humanoid, HumanoidStandup | **[-0.4, 0.4]** |

Always: `max_action = float(env.action_space.high[0])` — never hardcode.

## Required Wrappers (exact order matters)

```python
import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v5")
env = gym.wrappers.ClipAction(env)                    # 1. Safety net
env = gym.wrappers.NormalizeObservation(env)           # 2. Running mean/std
env = gym.wrappers.TransformObservation(env,           # 3. Clip outliers
    lambda obs: np.clip(obs, -10, 10))
```

Why: ClipAction first (safety), NormalizeObservation second (MuJoCo obs unbounded), TransformObservation clips outliers that would corrupt running stats.

## For TD3 Specifically
- No NormalizeReward — TD3 learns raw Q-values, normalizing hurts convergence
- No FrameStack — MuJoCo state includes velocities, already Markovian
- No vectorization needed — TD3 is off-policy, single env is standard
