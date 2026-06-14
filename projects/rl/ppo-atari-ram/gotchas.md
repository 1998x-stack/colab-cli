# PPO Atari RAM — Gotchas

## RAM observations have no spatial structure

Atari RAM is 128 bytes of unstructured memory state. MLPs treat it as a flat vector — there's no spatial inductive bias. This makes some games (Pong, Asterix) easily learnable and others (Montezuma's Revenge) impossible without RAM-specific feature engineering. Screen pixels (CNN-based PPO) would learn differently.

## AsyncVectorEnv must be spawned in main

`gym.vector.AsyncVectorEnv` spawns subprocesses. On Colab, if env creation happens outside `__main__` guard, the subprocesses re-import the module and create infinite recursion. Always wrap env creation in `if __name__ == "__main__":`.

## 4 envs × 128 steps = 512 samples per PPO update

With 4 parallel envs each running 128 steps, each PPO update sees 512 transitions. Reducing num_envs below 4 makes GAE advantage estimation less stable. Increasing beyond 4 on Colab T4 doesn't help (VRAM for batch forward pass becomes bottleneck).

## Orthogonal init with gain=0.01 for actor head

Standard orthogonal init (gain=1.0) produces too-large initial logits for discrete action spaces. The actor head uses gain=0.01 to start with near-uniform action distribution. Without this, the agent commits to bad policies in the first few updates.

## Config names use hyphens, env IDs use slashes

Config filenames: `ALE-Pong-v5.json` (hyphens). Gym env IDs: `ALE/Pong-v5` (slashes). The config loader translates between them. Don't rename config files without updating the translation.

## Solved thresholds vary widely

| Game | Solved threshold | Typical convergence |
|------|-----------------|-------------------|
| Pong | 18 (avg reward) | ~200K steps |
| Asterix | 5000 | ~1M steps |
| MsPacman | 3000 | ~2M steps |

Pong converges fast and serves as a pipeline verification. Asterix and MsPacman need much more compute — on Colab free-tier T4 (~8 min GPU window), you'll only get through the warmup phase.

## Time budget on Colab T4

With 4 envs × 128 steps per rollout, each iteration processes ~512 frames. At ~3000 frames/sec (MLP on T4 RAM), one iteration takes ~0.17s + PPO update (~0.05s) = ~0.22s. In 8 minutes you can do ~2200 iterations or ~1.1M frames — enough for Pong but not MsPacman.
