# DQN Atari Pong

Deep Q-Network with dueling architecture on ALE/Pong-v5. Uses a CNN encoder over stacked grayscale frames, experience replay, a target network, and epsilon-greedy exploration.

## Usage

```bash
# Local training
python train.py

# Colab deployment
cb launch.py [--args]
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Environment | ALE/Pong-v5 |
| Frame stack | 4 |
| Frame size | 84x84 grayscale |
| Replay buffer size | 100,000 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| Target network update | Every 1,000 steps |
| Epsilon decay | 1.0 to 0.01 over 50,000 steps |
| Max episodes | 500 |
| Learn frequency | Every 4 frames |
| Loss | Smooth L1 (Huber) |
| Gradient clipping | 10 |
| Solved threshold | Avg100 return > 18 |

## Network architecture

Dueling DQN with CNN encoder:
- Conv2d(4, 32, kernel=8, stride=4) → ReLU
- Conv2d(32, 64, kernel=4, stride=2) → ReLU
- Conv2d(64, 64, kernel=3, stride=1) → ReLU

Dueling streams from 3136-dim conv features:
- Value: Linear(3136, 256) → ReLU → Linear(256, 1)
- Advantage: Linear(3136, 256) → ReLU → Linear(256, action_dim)

Q(s,a) = V(s) + A(s,a) - mean(A(s))

Frame preprocessing: RGB (210x160x3) → grayscale → resize 84x84 → normalize to [0,1].

## Gotchas

- Environment is hardcoded to `ALE/Pong-v5` (mode=0, difficulty=0).
- Requires `opencv-python-headless` for frame resize/cvtColor.
- System deps `libcairo2-dev` and `libpango1.0-dev` needed for ALE rendering.
- Supports resume from checkpoint: scans output directory for latest `.pt` file.
- Early termination when avg100 return exceeds 18 (Pong solved condition).
