# MuJoCo Gymnasium Environments — Complete Reference

Data collected from gymnasium 1.3.0 (Colab, 2026-06-14). 11 environments, 39 versioned IDs (v2–v5).

## Quick Reference Table (v5)

| Env | Obs | Act | Action Range | Steps/Ep | Reward Threshold | FPS | Difficulty |
|-----|-----|-----|-------------|----------|------------------|-----|-----------|
| InvertedPendulum-v5 | 4 | 1 | [-3.0, 3.0] | 1000 | 950.0 | 25 | ★☆☆ |
| InvertedDoublePendulum-v5 | 9 | 1 | [-1.0, 1.0] | 1000 | 9100.0 | 20 | ★★☆ |
| Reacher-v5 | 10 | 2 | [-1.0, 1.0] | 50 | -3.75 | 50 | ★★☆ |
| Swimmer-v5 | 8 | 2 | [-1.0, 1.0] | 1000 | 360.0 | 25 | ★★☆ |
| Pusher-v5 | 23 | 7 | [-2.0, 2.0] | 100 | None | 20 | ★★☆ |
| Hopper-v5 | 11 | 3 | [-1.0, 1.0] | 1000 | 3800.0 | 125 | ★★★ |
| HalfCheetah-v5 | 17 | 6 | [-1.0, 1.0] | 1000 | 4800.0 | 20 | ★★★ |
| Walker2d-v5 | 17 | 6 | [-1.0, 1.0] | 1000 | None | 125 | ★★★ |
| Ant-v5 | 105 | 8 | [-1.0, 1.0] | 1000 | 6000.0 | 20 | ★★★★ |
| Humanoid-v5 | 348 | 17 | [-0.4, 0.4] | 1000 | None | 67 | ★★★★★ |
| HumanoidStandup-v5 | 348 | 17 | [-0.4, 0.4] | 1000 | None | 67 | ★★★★★ |

## Environment Categories

### Locomotion (5 envs) — forward movement reward

Forward-running agents. Reward = forward velocity minus control cost. The core MuJoCo benchmark suite.

#### HalfCheetah-v5
- **Task**: 2D cheetah runs forward as fast as possible
- **Obs (17)**: joint positions (8), joint velocities (8), root z-height (1)
- **Act (6)**: torque on 6 leg joints
- **Solved**: avg reward ≥ 4800 over 100 episodes
- **Notes**: Easiest locomotion env. DDPG reaches ~4000 in 200 episodes. SAC typically 5000+ in 100k steps.

#### Hopper-v5
- **Task**: 2D one-legged robot hops forward without falling
- **Obs (11)**: joint positions (5), joint velocities (5), root height + touch sensor (1)
- **Act (3)**: torque on thigh, leg, foot joints
- **Solved**: avg reward ≥ 3800 over 100 episodes
- **Notes**: Sparse failure signal (falling terminates episode). Tricky for deterministic policies — DDPG often collapses. TD3/SAC handle it better.

#### Walker2d-v5
- **Task**: 2D bipedal walker moves forward stably
- **Obs (17)**: joint positions (8), joint velocities (8), root height (1)
- **Act (6)**: torque on 6 leg joints
- **Notes**: No official solved threshold. Gait emergence is the key signal — the walker should develop a natural stride.

#### Ant-v5
- **Task**: 3D quadruped ant runs forward
- **Obs (105)**: joint positions (13) + velocities (13) — plus contact forces (52) and torso orientation (27) that v4 didn't have
- **Act (8)**: torque on 8 leg joints
- **Solved**: avg reward ≥ 6000 over 100 episodes
- **Notes**: v5 observation space is **3.9× larger** than v4 (105 vs 27). The extra dimensions are contact force sensors and more detailed torso state. High-dimensional action space (8D) makes Q-function learning harder — SAC recommended.

#### Swimmer-v5
- **Task**: N-link swimmer in 2D fluid, moves forward via body undulation
- **Obs (8)**: joint positions (4), joint velocities (4)
- **Act (2)**: torque on 2 rotor joints
- **Solved**: avg reward ≥ 360 over 100 episodes
- **Notes**: Simplest locomotion env (low dim, easy exploration). Good first test after InvertedPendulum. Small network (mlp-small) sufficient.

### Manipulation (2 envs) — target reaching with sparse reward

#### Reacher-v5
- **Task**: 2-joint arm reaches a random target with its fingertip
- **Obs (10)**: arm joint states (4), target position (2), fingertip-to-target vector (2), fingertip position (2)
- **Act (2)**: torque on 2 arm joints
- **Solved**: avg reward ≥ -3.75 (negative reward — closer to 0 is better)
- **Max steps**: 50 (very short — plan accordingly)
- **Notes**: Sparse distance-based reward. v5 observation dropped 1 dimension vs v4 (11→10). The 50-step limit means you need many episodes — 500k timesteps = 10,000 episodes.

#### Pusher-v5
- **Task**: 7-DOF arm pushes a puck to a target, using a "hand" paddle
- **Obs (23)**: arm joint states (14), object state (6), goal position (3)
- **Act (7)**: torque on 7 arm joints
- **Solved**: no official threshold
- **Max steps**: 100
- **Notes**: Pusher-v4 is broken with mujoco≥3 — must use v5. Action bounds are [-2, 2], wider than most envs. Reward is sparse (distance-based).

### Balance (2 envs) — high-dimensional bipedal stability

#### Humanoid-v5
- **Task**: 3D humanoid walks forward without falling
- **Obs (348)**: full-body joint states, contact forces, center-of-mass, inertia
- **Act (17)**: torque on 17 joints
- **Action bounds**: [-0.4, 0.4] — narrower than [-1,1], critical for tanh-squashed policies
- **Notes**: v5 observation is 348 (vs 376 in v4) — removed redundant sensors. 17D action space is the largest of any MuJoCo env. Needs large network (mlp-large or deeper) and 2M+ timesteps. Colab T4 can handle batch size 64. SAC with auto-tuned entropy is the recommended algorithm.

#### HumanoidStandup-v5
- **Task**: Humanoid starts lying down, must stand up
- **Obs (348)**: same as Humanoid
- **Act (17)**: same torque limits [-0.4, 0.4]
- **Notes**: Sparse reward — agent only gets reward when upright. Very hard exploration problem. Needs significant random exploration (10k+ random steps) or reward shaping. Expected to take 2-5M timesteps.

### Simple (2 envs) — pendulum stabilization, 1D action

#### InvertedPendulum-v5
- **Task**: Balance a pole on a moving cart
- **Obs (4)**: cart position, cart velocity, pole angle, pole angular velocity
- **Act (1)**: horizontal force on cart
- **Solved**: avg reward ≥ 950 over 100 episodes
- **Notes**: The "hello world" of continuous control. Action range is [-3, 3] — wider than almost all other envs. 50k steps should solve it. Any algorithm works.

#### InvertedDoublePendulum-v5
- **Task**: Balance two linked pendulums on a cart (chaotic system)
- **Obs (9)**: cart state (2), pole1 angle+velocity, pole2 angle+velocity (4), plus 3 extra state vars in v5
- **Act (1)**: horizontal force on cart
- **Solved**: avg reward ≥ 9100 over 100 episodes
- **Notes**: v5 observation shrunk from 11→9 (removed redundant measurements). Harder than single pendulum — chaotic dynamics, narrow stability region. SAC or PPO recommended.

## v4 → v5 Changes

Dimensions that changed between v4 and v5 (gymnasium 1.x, mujoco≥3):

| Env | v4 Obs | v5 Obs | v4 Act | v5 Act | Change |
|-----|--------|--------|--------|--------|--------|
| **Ant** | 27 | **105** | 8 | 8 | +78 obs (contact forces + torso detail added) |
| **Humanoid** | 376 | **348** | 17 | 17 | -28 obs (redundant sensors removed) |
| **HumanoidStandup** | 376 | **348** | 17 | 17 | -28 obs |
| **InvertedDoublePendulum** | 11 | **9** | 1 | 1 | -2 obs |
| **Reacher** | 11 | **10** | 2 | 2 | -1 obs |

Unchanged: HalfCheetah (17→17), Hopper (11→11), Walker2d (17→17), InvertedPendulum (4→4), Swimmer (8→8)

Action dimensions never changed — only observation spaces.

**Pusher-v4 is broken** — calling `gym.make("Pusher-v4")` with mujoco≥3 raises an error. Use v5.

## Version Availability

Some envs skip v3 (no HumanoidStandup-v3, no InvertedPendulum-v3, etc.):

```
Ant:                  v2, v3, v4, v5
HalfCheetah:          v2, v3, v4, v5
Hopper:               v2, v3, v4, v5
Humanoid:             v2, v3, v4, v5
HumanoidStandup:      v2,      v4, v5
InvertedDoublePendulum: v2,    v4, v5
InvertedPendulum:     v2,      v4, v5
Pusher:               v2,      v4, v5
Reacher:              v2,      v4, v5
Swimmer:              v2, v3, v4, v5
Walker2d:             v2, v3, v4, v5
```

## Obs/Action Space Properties

- **All observation spaces**: `Box(low=-inf, high=inf)` — unbounded. Always use `NormalizeObservation` wrapper.
- **All action spaces**: `Box(low=X, high=Y)` — bounded, but bounds vary. Never assume [-1, 1].
- **No env has `reward_range` set** — all return `None`.

### Action Bounds (non-standard)

Most envs use [-1, 1] but exceptions exist:

| Env | Low | High | Note |
|-----|-----|------|------|
| InvertedPendulum-v5 | -3.0 | 3.0 | 3× wider than typical |
| Humanoid-v5 | -0.4 | 0.4 | Narrow — tanh output × 0.4 |
| HumanoidStandup-v5 | -0.4 | 0.4 | Same |
| Pusher-v5 | -2.0 | 2.0 | 2× wider |
| All others | -1.0 | 1.0 | Standard |

**Always do**: `max_action = float(env.action_space.high[0])` and scale network output by this value. Don't hardcode `* 1.0`.

## Training Time Estimates (Colab T4 GPU)

Off-policy (SAC/TD3), conservative per-step estimate:

| Env | Steps to Solve | Time @ 0.5ms/step | Fits 10min Colab? |
|-----|---------------|-------------------|-------------------|
| InvertedPendulum-v5 | 50k | ~25s | Yes |
| Swimmer-v5 | 100k | ~50s | Yes |
| Reacher-v5 | 100k | ~50s | Yes |
| InvertedDoublePendulum-v5 | 100k | ~50s | Yes |
| Hopper-v5 | 200k | ~100s | Yes |
| HalfCheetah-v5 | 200k | ~100s | Yes |
| Walker2d-v5 | 200k | ~100s | Yes |
| Pusher-v5 | 300k | ~150s | Yes |
| Ant-v5 | 500k | ~300s | Yes |
| Humanoid-v5 | 2M | ~1000s | **No** — use Kaggle |
| HumanoidStandup-v5 | 2M+ | ~1200s+ | **No** — use Kaggle |

On-policy (PPO): multiply by ~5× (lower sample efficiency).

## Recommended Algorithm by Difficulty

| Tier | Envs | Recommended | Why |
|------|------|------------|-----|
| Trivial | InvertedPendulum | Any | All algorithms solve it quickly |
| Easy | Swimmer, Reacher | SAC | Sample-efficient, auto-tuned |
| Medium | Hopper, HalfCheetah, Walker2d, Pusher | SAC or TD3 | SAC more robust, TD3 simpler to implement |
| Hard | Ant, InvertedDoublePendulum | SAC | High-dim obs/act needs stable Q-learning |
| Extreme | Humanoid, HumanoidStandup | SAC | Auto-entropy tuning handles the exploration problem |

DDPG is not recommended for anything beyond InvertedPendulum — catastrophic forgetting makes it unreliable. PPO works on all envs but needs more wall-clock time (lower sample efficiency).

## Standard Wrappers

```python
import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v5")
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(
    env, lambda obs: np.clip(obs, -10, 10))
```

`ClipAction` — ensures actions stay within bounds (safety net for policy output).

`NormalizeObservation` — running mean/std normalization. Essential since observations are unbounded.

`TransformObservation(..., clip to [-10,10])` — prevents outlier observations from destabilizing the running stats.

## Data Source

Full 50KB structured JSON at `tmp/mujoco_env_reference.json` (from gymnasium 1.3.0 on Colab, 2026-06-14). Contains per-env: full obs/act bounds arrays, metadata (render_modes, render_fps), spec (max_episode_steps, reward_threshold), v4 vs v5 comparison table.
