# Atari Environments — Quick Reference

Full specs at `docs/reference/atari-environments.md`.

## Architecture Decision: Pixels vs RAM

| Mode | Obs Shape | Network | FPS (T4) | When to Use |
|------|-----------|---------|----------|-------------|
| Pixel | (4, 84, 84) after wrappers | CNN | ~200 | Final results, paper benchmarks |
| RAM | (128,) | MLP | ~800 | Quick experiments, algorithm debugging |

RAM mode is 4× faster and works with simple MLPs. Start with RAM, switch to pixels for final runs.

## Standard Wrapper Stack

```python
import ale_py  # MUST come before gym.make()

env = gym.make("ALE/Pong-v5",
    frameskip=1,                     # we handle frame skip via wrapper
    repeat_action_probability=0.0,   # deterministic
    full_action_space=False)         # minimal action set (3 for Pong, not 18)

env = gym.wrappers.AtariPreprocessing(env,
    screen_size=84,
    grayscale_obs=True,
    frame_skip=4,
    noop_max=30)

env = gym.wrappers.FrameStackObservation(env, 4)
# Output: (4, 84, 84)
```

## Action Space: `full_action_space` Matters

| Setting | Actions | When |
|---------|---------|------|
| `full_action_space=False` | 3–18 (game-specific minimum) | **Always use this** |
| `full_action_space=True` | 18 for every game | Only for compatibility testing |

The Atari 2600 joystick has 18 positions. Most games only use a subset. `full_action_space=False` gives the minimal viable set, which makes learning much faster.

## RAM Mode (no separate env ID)

```python
env = gym.make("ALE/Pong-v5", obs_type="ram")
# obs: (128,) uint8 — NO AtariPreprocessing needed
```

`ALE/Pong-ram-v5` does NOT exist. RAM is accessed via the `obs_type` parameter.

## Game Selection by Genre

| Genre | Easy (start here) | Hard (benchmark) |
|-------|-------------------|-------------------|
| Paddle | Pong (3 acts) | Breakout (4 acts) |
| Shooter | SpaceInvaders (6) | Seaquest (18), Gravitar (18) |
| Platformer | Frogger (4) | MontezumaRevenge (18) |
| Racing | Freeway (3) | Enduro (9) |
| Puzzle | Qbert (6) | Tetris (5) |

**MontezumaRevenge is the hardest Atari game** — needs curiosity-driven exploration (RND, ICM, Go-Explore). Standard DQN/PPO gets 0 reward.

## Training Scale

- DQN: 10M frames (~7 hours on T4)
- PPO: 10M frames (~7 hours on T4)
- RAM-mode smoke test: 500k frames (~20 min on T4)

Atari training doesn't fit Colab's ~10 minute GPU window. Use Kaggle (30h/week P100).

## 104 Games Quick Reference

**Action distribution**: 52 games use 18 actions. 14 games use 6. 8 games use 9. 7 games use 10. 6 games use 4. Smaller counts for 3, 5, 7, 8, 12, 14, 16.

**Genres**: Shooter (46), Puzzle (27), Other (13), Platformer (6), Board/Card (6), Adventure (3), Racing (3), Paddle (3).
