# TD3 on Gymnasium Continuous Control

Twin Delayed DDPG (TD3) on Gymnasium continuous-control environments, implemented in PyTorch with GPU support. Adds three key improvements over DDPG: twin critics (min overestimation), delayed policy updates, and target policy smoothing.

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
| Best eval mean reward | -71.76 (episode 200) |
| Final eval mean reward | -71.76 (episode 200) |
| Total env steps | 40,000 |
| Network | Actor (256x256) + Twin Critics (256x256 each) |
| Batch size | 100 |
| Actor learning rate | 3e-4 |
| Critic learning rate | 3e-4 |
| Policy delay | 2 (actor updates every 2 critic steps) |
| Policy noise | 0.2 (clipped to 0.5) |
| Exploration noise | 0.1 (Gaussian) |
| Replay buffer | 100,000 transitions |
| Gradient clipping | 1.0 |

## Gotchas

- The policy noise used for target smoothing is separate from the exploration noise added during action selection.
- The critic network contains two independent Q-function architectures (Q1 and Q2) with separate parameters.
- Target networks are updated via soft updates (polyak averaging) with tau=0.005.
- Checkpoints and evaluation plots are saved every 50 episodes.
