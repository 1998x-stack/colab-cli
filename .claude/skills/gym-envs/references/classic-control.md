# Classic Control Environments — Quick Reference

Full specs at `docs/reference/classic-control-environments.md`.

## All 5 Environments

| Env | Obs | Act | Type | Max Steps | Solved |
|-----|-----|-----|------|-----------|--------|
| CartPole-v1 | 4 | 2 | Discrete | 500 | 475 (official), 195 (literature) |
| Acrobot-v1 | 6 | 3 | Discrete | 500 | -100 |
| MountainCar-v0 | 2 | 3 | Discrete | 200 | -110 |
| MountainCarContinuous-v0 | 2 | 1 | Continuous [-1,1] | 999 | 90 |
| Pendulum-v1 | 3 | 1 | Continuous [-2,2] | 200 | None (~-150 typical) |

## Purpose

Classic Control envs are **debugging tools**, not research benchmarks. They solve in <2 minutes on CPU. Use them to:

1. Verify a new algorithm implementation works before scaling to MuJoCo
2. Test reward shaping or exploration strategies quickly
3. Check for catastrophic forgetting (train DDPG on Pendulum — if it forgets, your implementation is correct)

## Algorithm Test Sequence

1. **CartPole** (DQN) — simplest. If DQN can't solve this, something is fundamentally broken.
2. **MountainCar** (DQN + ε-greedy decay) — tests exploration. Without ε decay, never solves.
3. **Pendulum** (TD3/SAC) — tests continuous control. DDPG will show catastrophic forgetting here.
4. **Acrobot** (DQN) — sparse reward, slightly harder exploration.

## Key Gotchas

- **CartPole-v1 not v0**: v0 had 200 max steps, v1 has 500. Use v1.
- **MountainCarContinuous reward ≠ MountainCar reward**: Discrete is -1/step. Continuous is 100 - action_penalty for reaching goal. Completely different functions.
- **Pendulum action is [-2,2]**: Wider than most envs. Don't clip to [-1,1].
- **Act vs Obs types**: CartPole/Acrobot/MountainCar have Discrete actions; MountainCarContinuous/Pendulum have Box actions. Check `env.action_space` before writing your policy.

## No Wrappers Required

Classic Control envs don't need NormalizeObservation or ClipAction — their dynamics already bound observations naturally. Just `gym.make()` and go.
