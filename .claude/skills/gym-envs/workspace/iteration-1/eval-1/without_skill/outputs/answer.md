# HalfCheetah-v5: TD3 Setup

## Wrappers needed

```python
import gymnasium as gym
from gymnasium.wrappers import ClipAction, RescaleAction, RecordEpisodeStatistics

env = gym.make("HalfCheetah-v5")
env = RescaleAction(env, -1.0, 1.0)   # map policy [-1,1] to native range
env = ClipAction(env)                  # clip after exploration noise
env = RecordEpisodeStatistics(env)     # track returns
```

Do NOT use NormalizeObservation/NormalizeReward — those belong in the algorithm's internal stats.

## Obs/Act Dimensions

- Observation: Box(-inf, inf, (17,))
- Action: Box(-inf, inf, (6,)) — unbounded in v5

## Action Range: NOT always [-1, 1]

v5 changed action spaces to unbounded:

| Version | Action Space |
|---------|-------------|
| v4 | Box(-1.0, 1.0, (6,)) |
| v5 | Box(-inf, inf, (6,)) |

In v5, MuJoCo actuator bounds were removed. The simulation has no native torque clamping. This is why RescaleAction(env, -1.0, 1.0) is essential — it re-bounds the action to the v4 convention.

Without this wrapper, exploration noise can produce torques that crash the simulator with NaN states.
