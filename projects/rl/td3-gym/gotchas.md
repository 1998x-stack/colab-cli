# TD3-Gym — Gotchas

## Policy noise ≠ exploration noise

TD3 uses two different noise types:
- **Target policy smoothing** (0.2, clipped to 0.5): Added to target actions during critic update. Prevents the policy from exploiting Q-function errors on narrow peaks.
- **Exploration noise** (0.1, Gaussian): Added to actions during rollout. Simpler than DDPG's OU noise.

Don't confuse the two — removing policy smoothing defeats TD3's main advantage over DDPG.

## Delayed policy updates: d=2

Actor + target networks updated every 2 critic updates. This lets the critics converge between policy changes, reducing the tendency to exploit overestimated Q-values. Setting d=1 degrades to DDPG-like behavior with twin critics only.

## Twin critics (min Q) is the core innovation

Using `min(Q1, Q2)` for the target value deliberately underestimates Q, counteracting DDPG's overestimation bias. Both critics are trained independently on the same data — the min operation happens only in the target computation.

## Pendulum solved at ~-200 (not 0)

Same as DDPG — the reward ranges from ~-1600 (worst) to 0 (perfect). TD3 typically reaches -100 to -70, better than DDPG's -165. The improvement comes from reduced overestimation (twin critics) and target smoothing.

## HalfCheetah/Hopper/Walker2d need separate launch

The default env is Pendulum-v1. For MuJoCo envs, pass `--env HalfCheetah-v4`. But td3-gym is designed for single-env training (unlike ddpg-td3-mujoco which runs all 6 combos). Use ddpg-td3-mujoco for multi-env comparison.

## Gradient clipping (1.0) applied to both actor and critics

Without clipping, the twin critics can diverge independently — one critic's gradient explosion doesn't affect the other, but the actor sees min(Q1, Q2) which can still be wrong. Clipping prevents this cascade.
