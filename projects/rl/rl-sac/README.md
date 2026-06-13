# SAC on MountainCarContinuous

Soft Actor-Critic (SAC) with automatic entropy tuning on MountainCarContinuous-v0, implemented in PyTorch with GPU support.

## Usage

```bash
# Local training (500 episodes, MountainCarContinuous-v0)
python sac_mountaincar.py

# Colab deployment
cb launch.py [--args]
```

## Key results

| Metric | Value |
|--------|-------|
| Environment | MountainCarContinuous-v0 |
| Episodes | 500 |
| Batch size | 256 |
| Replay buffer | 1,000,000 transitions |
| Learning rate | 3e-4 (actor, critic, and alpha) |
| Hidden layers | 256 x 2 |
| Target entropy | -act_dim (automatic tuning) |
| Start steps | 10,000 (uniform random) |
| Soft update tau | 0.005 |

## Gotchas

- Uses a Gaussian policy with tanh squashing and reparameterized sampling (rsample).
- Automatic entropy tuning via learnable `log_alpha` parameter (target entropy = -action_dim).
- Checkpoints saved to `/content/checkpoints/` and automatically resumed from latest `.pt` file.
- First 10,000 steps use uniform random action selection (no policy) for exploration.
