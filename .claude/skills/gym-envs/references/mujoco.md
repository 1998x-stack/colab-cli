# MuJoCo Environments — Quick Reference

Full specs at `docs/reference/mujoco-environments.md`. This file covers gotchas and selection guidance.

## Selection by Task

| Task | Env | Obs | Act | Solved Threshold | Time on T4 |
|------|-----|-----|-----|-----------------|------------|
| Quickest test | InvertedPendulum-v5 | 4 | 1 | 950 | ~25s |
| Easy locomotion | Swimmer-v5 | 8 | 2 | 360 | ~50s |
| Standard benchmark | HalfCheetah-v5 | 17 | 6 | 4800 | ~100s |
| Bipedal | Hopper-v5 | 11 | 3 | 3800 | ~100s |
| Gait learning | Walker2d-v5 | 17 | 6 | None | ~100s |
| High-dim | Ant-v5 | 105 | 8 | 6000 | ~300s |
| Extreme | Humanoid-v5 | 348 | 17 | None | ~1000s |

## Action Bounds (critical — never assume [-1,1])

| Env | Action Range |
|-----|-------------|
| InvertedPendulum-v5 | **[-3.0, 3.0]** |
| Humanoid-v5 | **[-0.4, 0.4]** |
| HumanoidStandup-v5 | **[-0.4, 0.4]** |
| Pusher-v5 | **[-2.0, 2.0]** |
| All others | [-1.0, 1.0] |

Always: `max_action = float(env.action_space.high[0])`

## v4 → v5 Obs Dimension Changes

| Env | v4 Obs | v5 Obs |
|-----|--------|--------|
| Ant | 27 | **105** |
| Humanoid | 376 | **348** |
| HumanoidStandup | 376 | **348** |
| InvertedDoublePendulum | 11 | **9** |
| Reacher | 11 | **10** |

HalfCheetah, Hopper, Walker2d, InvertedPendulum, Swimmer unchanged.

## Algorithm Recommendations

| Tier | Envs | Algorithm |
|------|------|-----------|
| Easy | InvertedPendulum, Swimmer | Any (debug here first) |
| Medium | HalfCheetah, Hopper, Walker2d, Pusher, Reacher | SAC or TD3 |
| Hard | Ant, InvertedDoublePendulum | SAC |
| Extreme | Humanoid, HumanoidStandup | SAC (use Kaggle, not Colab) |

DDPG has catastrophic forgetting on all MuJoCo envs — TD3 or SAC preferred.
PPO works but needs 4+ vectorized envs and more wall-clock time.

## Standard Wrapper Stack

```python
env = gym.make(env_id)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
```

Order matters: ClipAction → NormalizeObservation → TransformObservation.
