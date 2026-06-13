# DDPG on Gymnasium Continuous Control

Deep Deterministic Policy Gradient (DDPG) with OU noise on Gymnasium continuous-control environments, implemented in PyTorch with GPU support.

## Usage

```bash
# Local training (default: Pendulum-v1, 200 episodes)
python train.py [--env Pendulum-v1] [--episodes 200] [--seed 42]

# Colab deployment
cb launch.py [--args]
```

## Key results

Default environment is **Pendulum-v1** (200 episodes, 200 steps/episode, CUDA).

| Metric | Value |
|--------|-------|
| Best eval mean reward | -40.95 (episode 40) |
| Final eval mean reward | -165.96 (episode 200) |
| Total env steps | 40,000 |
| Network | Actor + Critic (256x256 each) |
| Batch size | 64 |
| Learning rate | 1e-3 (actor & critic) |
| Exploration | OU noise (theta=0.15, sigma=0.2) |
| Replay buffer | 100,000 transitions |
| Gradient clipping | 1.0 |

## Gotchas

- Output directory defaults to `/content/ddpg-output/` (Colab path). Override with `--out_dir` for local runs.
- OU noise is annealed linearly during the warmup phase (1000 steps) and clipped to action bounds.
- The logger writes to both stdout and `train.log` simultaneously.
