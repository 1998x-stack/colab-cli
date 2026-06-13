# DDQN vs NoisyNet

Double DQN and NoisyNet (DQN with factorized Gaussian noise) comparison on three discrete-action Gymnasium environments. Uses LayerNorm, per-environment hyperparameters, and Prioritized Experience Replay for sparse-reward envs.

## Usage

```bash
# Local training
python train.py [--envs CartPole-v1 MountainCar-v0 Acrobot-v1] [--algos ddqn noisy]

# Colab deployment
cb launch.py [--args]
```

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| Buffer size | 50,000 |
| Batch size | 128 |
| Gamma | 0.99 |
| Tau (soft update) | 0.005 |
| Epsilon range | 1.0 to 0.02 |
| PER alpha / beta | 0.6 / 0.4 |
| Optimizer | Adam |
| Loss | Smooth L1 |

### Per-environment configs

| Env | Episodes | LR | Target update | Warmup | PER |
|-----|----------|----|---------------|--------|-----|
| CartPole-v1 | 300 | 5e-3 | 50 | 500 | No |
| MountainCar-v0 | 500 | 1e-3 | 100 | 2,000 | Yes |
| Acrobot-v1 | 500 | 1e-3 | 100 | 2,000 | Yes |

### Networks

- **DDQN**: MLP(obs_dim, 128) → LayerNorm → ReLU → MLP(128, 64) → LayerNorm → ReLU → MLP(64, n_actions). Epsilon-greedy exploration with exponential decay.
- **NoisyNet**: Same architecture but every Linear replaced with `NoisyLinear` using factorized Gaussian noise (sigma=0.5). No epsilon-greedy — exploration is intrinsic to the noise.

DDQN uses Double DQN target: `r + gamma * Q_target(s', argmax Q_online(s'))`.

## Gotchas

- NoisyLinear samples new noise at the start of each episode and before each training step.
- Prioritized replay uses a binary SumTree with capacity rounded up to the next power of two.
- Indexes from the SumTree sample are wrapped modulo the deque length for capacity mismatch.
- This project compares algorithms on simple Gymnasium environments (CartPole, MountainCar, Acrobot) — not Atari RAM.
