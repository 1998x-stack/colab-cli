---
name: gym-envs
description: >
  Use when working with Gymnasium (Gym) environments — discovering, configuring,
  wrapping, or debugging any RL environment type. Triggers on mentions of Gym,
  Gymnasium, Classic Control, MuJoCo, Atari, Box2D, RL environments, env setup,
  env wrappers, vectorized envs, or when the user needs environment specs (obs/act
  dimensions, reward thresholds, action bounds). Also trigger when setting up a
  new RL training project and the user needs to choose or configure environments.
---

# Gymnasium Environments

Comprehensive guidance for working with Gymnasium environments across all environment types. Covers discovery, configuration, wrapping, vectorization, and advanced patterns.

## Environment Landscape

Gymnasium provides four major environment families:

| Family | Obs Type | Act Type | Count | Use Case |
|--------|---------|----------|-------|----------|
| Classic Control | Low-dim vectors (2–6) | Discrete or Continuous(1) | 5 | Quick algorithm tests, debugging |
| MuJoCo | Low-dim vectors (4–348) | Continuous (1–17) | 11 | Continuous control benchmarks |
| Atari | Pixels (210×160×3) or RAM (128) | Discrete (3–18) | 104 | DQN/PPO benchmarks, pixel-based RL |
| Box2D | Low-dim + pixels mixed | Discrete or Continuous | varies | 2D physics, lighter than MuJoCo |

The first decision is always: which family fits the task? Classic Control for debugging, MuJoCo for continuous control research, Atari for discrete pixel-based RL.

## Core Workflow

### 1. Discover an environment

Never hardcode obs/act dimensions. Read them from the live environment:

```python
import gymnasium as gym

env = gym.make("HalfCheetah-v5")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# For continuous actions — bounds vary per env, never assume [-1,1]
if hasattr(env.action_space, 'high'):
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

# For discrete actions
if hasattr(env.action_space, 'n'):
    n_actions = env.action_space.n

max_eps = env.spec.max_episode_steps  # may be None for Atari
threshold = env.spec.reward_threshold  # may be None
env.close()
```

Use the bundled `scripts/check_env.py` for a one-line spec dump:
```bash
python check_env.py HalfCheetah-v5       # single env
python check_env.py --all-mujoco         # all 11 MuJoCo envs
python check_env.py --all-classic        # all 5 Classic Control envs
```

### 2. Choose and apply wrappers

Wrappers are the primary configuration mechanism. Apply them in **this order** — each depends on the previous:

**MuJoCo / Continuous Control:**
```python
env = gym.make("HalfCheetah-v5")
env = gym.wrappers.ClipAction(env)                    # safety net
env = gym.wrappers.NormalizeObservation(env)           # running mean/std
env = gym.wrappers.TransformObservation(env,           # clip outliers
    lambda obs: np.clip(obs, -10, 10))
```

Why this order: ClipAction first ensures actions never violate the env's bounds regardless of what the policy outputs. NormalizeObservation is critical because MuJoCo obs are unbounded Box(-inf, inf). TransformObservation clips extreme values that would destabilize the running stats.

**Atari (pixel-based):**
```python
import ale_py  # registers ALE namespace — must precede gym.make()

env = gym.make("ALE/Pong-v5", frameskip=1,
               repeat_action_probability=0.0,
               full_action_space=False)   # minimal action set

env = gym.wrappers.AtariPreprocessing(env,
    screen_size=84, grayscale_obs=True, frame_skip=4,
    noop_max=30)                           # random no-ops at reset

env = gym.wrappers.FrameStackObservation(env, 4)
# Final obs: (4, 84, 84) — 4 stacked grayscale frames
```

Why: AtariPreprocessing handles frame-skip (action repeated 4×), max-pooling over 2 frames (eliminates sprite flicker), and rescaling. FrameStack(4) gives the CNN velocity information — the agent sees motion, not just static images.

**Atari (RAM-based, MLP instead of CNN):**
```python
env = gym.make("ALE/Pong-v5", obs_type="ram", full_action_space=False)
# obs: (128,) uint8 — no image preprocessing needed
```
RAM mode is ~4× faster than pixel mode and works with simple MLPs. Good for quick experiments on games where RAM contains position/score info. No separate env ID — it's a `gym.make()` parameter.

**Classic Control (minimal wrapping):**
```python
env = gym.make("CartPole-v1")  # no wrappers needed
# Optional: add TimeLimit for custom episode length
env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
```

### 3. Vectorize for parallel rollouts

Single environment → Vectorized environments. Required for PPO, beneficial for any on-policy algorithm:

```python
# SyncVectorEnv — simple, single-process
envs = gym.vector.SyncVectorEnv([
    lambda: make_env("HalfCheetah-v5", seed=i) for i in range(4)
])

# AsyncVectorEnv — multiprocessing, for CPU-heavy envs
envs = gym.vector.AsyncVectorEnv([
    lambda: make_env("HalfCheetah-v5", seed=i) for i in range(4)
])
```

Key difference: `obs` from vectorized envs is `(num_envs, *obs_shape)`, not `(*obs_shape,)`. Rewards/dones are arrays of length `num_envs`. Use `SyncVectorEnv` by default — AsyncVectorEnv adds overhead that only pays off with CPU-bound environments.

## Environment-Specific Guidance

### MuJoCo

Read `references/mujoco.md` for complete specs and gotchas. Key points:

- **Action bounds vary**: Most are [-1,1], but InvertedPendulum is [-3,3], Humanoid is [-0.4,0.4]. Always read `env.action_space.high[0]` — never hardcode.
- **Obs unbounded**: All MuJoCo observations are Box(-inf, inf). NormalizeObservation is not optional.
- **v5 is current**: Use v5 env IDs. v4 used different observation dimensions for Ant (27→105), Humanoid (376→348). Pusher-v4 is broken with mujoco≥3.
- **Training time**: 200k steps on T4 ≈ 100s for SAC. PPO needs ~5× more. Humanoid needs 2M+ steps → use Kaggle.

### Atari

Read `references/atari.md` for full game list and categories. Key points:

- **104 games**, all `ALE/GameName-v5`. Import `ale_py` before `gym.make()`.
- **No separate RAM envs**: Use `obs_type="ram"`, not `ALE/Pong-ram-v5` (doesn't exist).
- **Action space**: 3–18 discrete actions. `full_action_space=False` gives the minimal set.
- **No default max_episode_steps**: Atari envs run until game over. Add TimeLimit wrapper if needed.
- **Training scale**: DQN needs 10M+ frames (~7 hours on T4). Not suitable for Colab's ~10 min window.

### Classic Control

Read `references/classic-control.md` for full specs. Key points:

- **5 envs**: CartPole, Acrobot, MountainCar (discrete + continuous), Pendulum
- **Debugging tools**: These solve in <2 min on CPU. Use them to verify algorithm correctness before scaling to MuJoCo.
- **CartPole-v1** (not v0): v0 has 200 max steps, v1 has 500. Always use v1.
- **MountainCarContinuous** has a completely different reward function from discrete MountainCar.

## Advanced Patterns

### Custom wrapper

When one env needs behavior not in the standard wrapper library:

```python
class ClipObservation(gym.ObservationWrapper):
    """Clip observations to a fixed range. Use when you know the physics bounds."""
    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

    def observation(self, obs):
        return np.clip(obs, self.low, self.high)
```

### Recording episodes

```python
env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "videos/", episode_trigger=lambda ep: ep % 100 == 0)
```

### Seeding for reproducibility

```python
env = gym.make("HalfCheetah-v5")
obs, _ = env.reset(seed=42)  # seed on reset, not on make
```

For vectorized envs, use different seeds per env: `env.reset(seed=42 + i)` for env i.

### Action scaling for continuous policies

This is the most common MuJoCo bug. The policy outputs raw values (e.g. from `tanh`, range [-1,1]) that must be scaled to the env's action range:

```python
class Actor(nn.Module):
    def forward(self, obs):
        mean = self.mean_head(obs)
        return torch.tanh(mean) * self.max_action  # scale to env's range
```

Where `max_action = float(env.action_space.high[0])`. This value changes per env — never hardcode `* 1.0`.

### Environment compatibility check

Before deploying to Colab/Kaggle, verify the env is importable and dimensions match:

```python
import gymnasium as gym
env = gym.make(ENV_ID)
obs_sample, _ = env.reset()
assert obs_sample.shape == env.observation_space.shape, \
    f"Obs mismatch: {obs_sample.shape} vs {env.observation_space.shape}"
action_sample = env.action_space.sample()
obs, reward, terminated, truncated, _ = env.step(action_sample)
print(f"OK: obs={obs.shape}, reward={reward}, action={action_sample.shape}")
env.close()
```

## Gotchas (read before any implementation)

1. **Never hardcode obs/act dimensions.** Read from the live env. Config files rot — the env is authoritative.
2. **Action bounds are NOT uniform.** InvertedPendulum uses [-3,3], Humanoid uses [-0.4,0.4], Pusher uses [-2,2]. Scale your policy output accordingly.
3. **MuJoCo obs are unbounded.** Always use NormalizeObservation + clip to [-10,10].
4. **Atari RAM envs don't exist as separate IDs.** `ALE/Pong-ram-v5` returns 404. Use `gym.make("ALE/Pong-v5", obs_type="ram")`.
5. **Atari needs `import ale_py`** before any `gym.make("ALE/...")` call — the import registers the namespace.
6. **Pusher-v4 is broken** with mujoco≥3. Use v5.
7. **v4→v5 obs dims changed** for Ant (27→105), Humanoid (376→348), HumanoidStandup (376→348), InvertedDoublePendulum (11→9), Reacher (11→10).
8. **`env.reward_range` is on `env.unwrapped`, not on `env`** — wrappers like TimeLimit hide it.
9. **`gym.make()` with Atari `frameskip=1`** and use AtariPreprocessing for frame skip — don't rely on the env's built-in frameskip.
10. **PPO needs 4+ vectorized envs** for gradient stability. Single-env PPO on MuJoCo doesn't learn.

## Reference Files

- `references/mujoco.md` — All 11 MuJoCo v5 envs, full specs, v4→v5 diffs, training times
- `references/classic-control.md` — All 5 Classic Control envs, algorithms by difficulty
- `references/atari.md` — All 104 Atari games, genres, action distributions, wrapper recipes
- `references/wrappers.md` — Complete Gymnasium wrapper catalog with use cases
- `references/gotchas.md` — Full gotcha list with root causes and fixes

## Scripts

- `scripts/check_env.py` — Quick environment spec dumper. Supports single env or batch mode.
