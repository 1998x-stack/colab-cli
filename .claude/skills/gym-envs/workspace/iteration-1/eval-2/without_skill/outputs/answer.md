# Atari with MLP

Yes, you can use Atari with an MLP via RAM observations.

## Option 1: `PongRam-v5` (may need older gym)

```python
import gymnasium as gym
env = gym.make("PongRam-v5")  # obs: (128,) uint8
```

## Option 2: `ALE/Pong-ram-v5` (gymnasium 0.29+)

```python
env = gym.make("ALE/Pong-ram-v5")  # obs: (128,) uint8
```

## Normalization

Cast to float32, divide by 255.0:
```python
obs = obs.float() / 255.0
```

## Full Example

```python
import gymnasium as gym
import numpy as np

env = gym.make("PongRam-v5")
obs, _ = env.reset()
obs = obs.astype(np.float32) / 255.0

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    obs = obs.astype(np.float32) / 255.0
    if terminated or truncated:
        obs, _ = env.reset()
```
