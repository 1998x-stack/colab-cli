# Tabular SARSA on CartPole

On-policy tabular SARSA with linear-decay learning rate and epsilon-greedy exploration on CartPole-v1, using state discretization (12 bins per dimension = 12^4 = 20,736 states).

## Usage

```bash
# Local training (3000 episodes, CartPole-v1)
python train.py

# Colab deployment
cb launch.py [--args]
```

## Key results

| Metric | Value |
|--------|-------|
| Best avg100 reward | 350.41 (episode 2724) |
| Final avg100 reward | 276.26 (episode 3000) |
| Total training time | 59 seconds (CPU) |
| Eval episode rewards | 227, 316, 299, 211, 300 |
| Maximum episode reward | 500 (CartPole-v1 solved threshold) |
| Q-table size | 20,736 states x 2 actions |
| Learning rate | 0.5 → 0.01 (linear decay) |
| Epsilon | 1.0 → 0.01 (multiplicative decay 0.9985) |

## Gotchas

- Entirely CPU-based (NumPy); no GPU is needed or used for tabular RL.
- Two state dimensions (cart velocity, pole velocity at tip) are unbounded and clipped to [-3, 3] for discretization.
- The Q-table is a float32 NumPy array of shape (20736, 2), initialized to zeros.
- Checkpoints saved as `.npy` files every 500 episodes.
- Performance plateaus after ~2700 episodes due to discretization granularity.
