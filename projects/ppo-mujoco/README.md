# PPO MuJoCo

Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) on MuJoCo continuous-control environments. Uses a Gaussian policy (learned mean + log std) and synchronous vectorized environments.

## Usage

```bash
# Local training (default envs: HalfCheetah, Hopper, Walker2d)
python train.py [--envs HalfCheetah-v5 Hopper-v5 Walker2d-v5]

# Colab deployment
cb launch.py [--args]
```

## Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Total timesteps | 1,000,000 (per env) |
| Num vectorized envs | 1 |
| Num steps per rollout | 2,048 |
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip coefficient | 0.2 |
| Entropy coefficient | 0.01 |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |
| PPO epochs | 5 |
| Num minibatches | 32 |

## Network architectures

All networks share a trunk with a Gaussian actor head (mean vector + learned log std) and a scalar critic head.

| Key | Architecture |
|-----|-------------|
| `mlp-small` | [128, 128] hidden layers |
| `mlp-medium` | [256, 256] hidden layers |
| `mlp-large` | [512, 512] hidden layers |
| `resmlp` | [256, 256] with LayerNorm and residual skip connections |

Orthogonal initialization; actor head uses gain=0.01.

## Environments

| Env | Obs dim | Actions | Timesteps |
|-----|---------|---------|-----------|
| HalfCheetah-v5 | 17 | 6 | 1M |
| Hopper-v5 | 11 | 3 | 1M |
| Walker2d-v5 | 17 | 6 | 1M |
| Ant-v5 | 27 | 8 | 1M |
| Humanoid-v5 | 376 | 17 | 2M |

11 MuJoCo v5 environments supported via `generate_configs.py`.

## Gotchas

- Environments are wrapped with `ClipAction`, `NormalizeObservation`, and observation clipping to [-10, 10].
- Uses `gym.vector.SyncVectorEnv` (single-process, unlike the async vectorization in PPO Atari RAM).
- Diagonal Gaussian policy with state-independent log standard deviation.
- Checkpoints and PNG training curves saved to output directory per env.
