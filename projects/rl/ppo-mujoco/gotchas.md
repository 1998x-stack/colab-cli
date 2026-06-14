# PPO MuJoCo — Gotchas

## SyncVectorEnv (not Async) for MuJoCo

MuJoCo environments must run synchronously (`gym.vector.SyncVectorEnv`) because MuJoCo's physics engine is not fork-safe. AsyncVectorEnv spawns subprocesses that each initialize MuJoCo independently, causing segfaults on Colab/Kaggle.

## 1 env × 2048 steps = large rollouts

Unlike Atari RAM (4 envs × 128 steps), MuJoCo PPO uses 1 env × 2048 steps per rollout. This trades parallelism for longer trajectories — important for MuJoCo where credit assignment spans hundreds of steps. Total steps per update: 2048.

## Diagonal Gaussian policy with learned log std

The action distribution is parameterized as `N(mean(state), exp(log_std))` where `log_std` is a learnable parameter (not a function of state). This state-independent variance works well for MuJoCo. State-dependent variance (another network head) adds complexity without gains on standard benchmarks.

## ClipAction + NormalizeObservation are critical

Without `ClipAction`, the Gaussian policy can output values outside the action space, causing MuJoCo to return NaN. Without `NormalizeObservation`, the running mean/std of observations drifts and the policy collapses after ~100K steps.

## HalfCheetah solved at ~4000+ reward

v5 environments use the new Gymnasium API. HalfCheetah-v5 solved threshold is ~4000 (similar to v4). Hopper-v5 ~2500, Walker2d-v5 ~3000, Ant-v5 ~3000.

## 1M steps per env = 15-30 min on T4

MuJoCo PPO at 1M steps with 1 env processes ~2000 steps/sec on T4. Total: 1M/2000 = ~500s ≈ 8 min per env. Three envs = ~25 min — exceeds free-tier Colab GPU window. Run one env per session, or use Kaggle (P100, 30h/week).

## Humanoid-v5 needs 2M steps and more GPU

Humanoid has 376-dim observations and 17-dim actions. 2M steps at ~500 steps/sec = 66 min. VRAM usage is higher due to larger network + larger rollout batch. Free-tier T4 (15.6 GB) fits but the time budget doesn't.

## Observation clipping to [-10, 10]

NormalizeObservation can produce extreme values for rare states (e.g., robot fallen over). Clipping to [-10, 10] prevents these from creating large policy gradients that destabilize training.

## Orthogonal init with gain=0.01 for continuous control

Same as PPO Atari RAM — the actor head uses gain=0.01 to start with near-zero mean actions. Without it, the initial policy outputs extreme actions and the agent never recovers.
