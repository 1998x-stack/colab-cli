# Gymnasium Wrappers — Complete Catalog

Wrappers modify environment behavior without changing the environment itself. Apply in the order shown — each depends on the previous wrapper's output.

## Observation Wrappers

| Wrapper | What It Does | When to Use |
|---------|-------------|-------------|
| `NormalizeObservation` | Running mean/std normalization of observations | **Always for MuJoCo** (obs are unbounded). Not needed for Classic Control or RAM Atari. |
| `TransformObservation` | Apply arbitrary function to each observation | Clip to [-10,10] after NormalizeObservation to prevent stat explosion. |
| `GrayScaleObservation` | Convert RGB to grayscale | If you want grayscale without full AtariPreprocessing. Prefer AtariPreprocessing for Atari. |
| `ResizeObservation` | Resize observation to target shape | When you need a specific input size but not full preprocessing. |
| `FlattenObservation` | Flatten dict/tuple observations to 1D | Legacy compat. Modern envs use flat Box observations. |
| `FrameStackObservation` | Stack last N observations along new axis | **Required for Atari pixels** — gives CNN velocity info. Use 4 frames. |
| `FilterObservation` | Keep only specified keys from Dict obs | Multi-modal envs where you only need one modality. |
| `PixelObservationWrapper` | Convert state-based obs to pixels | When you need to train from pixels on a state-based env. |

## Action Wrappers

| Wrapper | What It Does | When to Use |
|---------|-------------|-------------|
| `ClipAction` | Clamp actions to env's bounds | **Always for MuJoCo** — safety net for policy output. |
| `RescaleAction` | Rescale action from one range to another | When policy output range ≠ env action range. Alternative to scaling in the network. |
| `TransformAction` | Apply arbitrary function to each action | Rare edge cases. Prefer ClipAction + network-side scaling. |
| `StickyAction` | Repeat previous action with probability p | Simulates hardware lag. Research use only. |

## Reward Wrappers

| Wrapper | What It Does | When to Use |
|---------|-------------|-------------|
| `ClipReward` | Clip reward to [min, max] | Atari DQN — clip to [-1, 1] for stable Q-learning. |
| `TransformReward` | Apply arbitrary function to reward | Reward shaping, scaling, sign flipping. |
| `NormalizeReward` | Running mean/std normalization of rewards | PPO — keeps advantage scale consistent across envs. |

## Episode Wrappers

| Wrapper | What It Does | When to Use |
|---------|-------------|-------------|
| `TimeLimit` | Truncate episode after N steps | **Atari** (no default max steps). Custom episode lengths. |
| `RecordEpisodeStatistics` | Track episode return, length, time | **Always in training** — gives per-episode metrics. |
| `RecordVideo` | Save MP4 videos of episodes | Qualitative eval — render every N episodes. |

## Atari-Specific

| Wrapper | What It Does | When to Use |
|---------|-------------|-------------|
| `AtariPreprocessing` | Frame-skip, max-pool, grayscale, resize, noop-reset | **Standard Atari pipeline**. Replaces 5+ individual wrappers. |

`AtariPreprocessing` combines: MaxAndSkipEnv (frame skip + flicker removal) + Grayscale + Resize + NoopResetEnv. Always use it instead of composing these manually.

## Standard Wrapper Chains

### MuJoCo (continuous control)
```
ClipAction → NormalizeObservation → TransformObservation(clip to [-10,10])
```

### Atari (pixel-based DQN/PPO)
```
AtariPreprocessing(screen_size=84, frame_skip=4, noop_max=30) → FrameStackObservation(4)
```

### Atari (RAM-based MLP)
```
(no wrappers needed — just obs_type="ram")
```

### Classic Control
```
(no wrappers needed)
```

### PPO on any env
```
[env-specific wrappers] → NormalizeReward → RecordEpisodeStatistics
```

## Custom Wrapper Pattern

```python
class MyWrapper(gym.Wrapper):
    """Modify step/observation/reward. Override only what you need."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # modify obs/reward/info here
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # modify initial obs
        return obs, info
```

Use ObservationWrapper/RewardWrapper/ActionWrapper subclasses when you only modify one thing — they auto-delegate the others to the base env.
