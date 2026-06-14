# Atari with MLP: Use RAM Mode

## The Answer

You can absolutely use Atari with an MLP. Use **RAM mode** — it gives a 128-dim vector observation instead of pixels.

## API: NOT a separate env ID

**Wrong** (does not exist in gymnasium 1.x):
```python
gym.make("ALE/Pong-ram-v5")   # ERROR: Environment not found
gym.make("Pong-ram-v5")       # ERROR: not registered
```

**Correct:**
```python
import ale_py  # MUST import before gym.make() — registers ALE namespace
import gymnasium as gym

env = gym.make("ALE/Pong-v5", obs_type="ram", full_action_space=False)
# obs: Box(128,) uint8 — 128 bytes of raw Atari 2600 RAM
```

## Key Details

- **Observation**: (128,) uint8 — all Atari RAM modes are the same size regardless of game
- **No wrappers needed** — no AtariPreprocessing, no FrameStack, no grayscale conversion
- **RAM content**: Game state (positions, scores, lives, enemy positions) — varies per game
- **Speed**: ~800 fps on T4 (vs ~200 fps for pixel mode with CNN)

## MLP Setup

```python
import ale_py, gymnasium as gym
import torch, torch.nn as nn

env = gym.make("ALE/Pong-v5", obs_type="ram", full_action_space=False)
# obs: (128,) uint8 → normalize to [0,1] float32

class MLPAgent(nn.Module):
    def __init__(self, obs_dim=128, n_actions=env.action_space.n, hidden=[128, 128]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.q_head = nn.Linear(in_dim, n_actions)

    def forward(self, obs):
        return self.q_head(self.trunk(obs.float() / 255.0))
```
