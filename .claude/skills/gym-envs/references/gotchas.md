# Gymnasium Environment Gotchas

Field-tested pitfalls, root causes, and fixes. Organized by impact.

## Critical (will silently break training)

### 1. Hardcoded action bounds
**Problem**: Assuming `max_action = 1.0` for all continuous envs. InvertedPendulum uses 3.0, Humanoid uses 0.4.
**Fix**: `max_action = float(env.action_space.high[0])` — read from env, never hardcode.
**Affects**: MuJoCo, any continuous control env.

### 2. MuJoCo observations unbounded → no NormalizeObservation
**Problem**: MuJoCo obs are Box(-inf, inf). Without normalization, the first few observations can be extreme values that break the policy network (NaN gradients).
**Fix**: Always wrap with NormalizeObservation + clip to [-10,10].
**Affects**: All MuJoCo envs.

### 3. PPO with 1 vectorized environment
**Problem**: PPO needs diverse rollouts for stable advantage estimation. Single-env PPO on MuJoCo produces noisy gradients — policy stays random (entropy ~8.5, reward flat).
**Fix**: Use `num_envs >= 4` with SyncVectorEnv or AsyncVectorEnv.
**Affects**: PPO on any env. SAC/TD3 are fine with 1 env.

### 4. Atari `import ale_py` missing
**Problem**: `gym.make("ALE/Pong-v5")` fails with "Namespace ALE not found" even though `gymnasium[atari]` is installed.
**Fix**: `import ale_py` before any `gym.make("ALE/...")` call. The import registers the namespace.
**Affects**: All Atari envs.

### 5. Atari RAM envs don't exist as separate IDs
**Problem**: `gym.make("ALE/Pong-ram-v5")` fails. The `-ram` suffix was removed in gymnasium 1.x.
**Fix**: `gym.make("ALE/Pong-v5", obs_type="ram")`. RAM is a mode, not an env.
**Affects**: All Atari RAM training.

## High Impact

### 6. v4→v5 obs dimension mismatch
**Problem**: Code written for v4 hardcodes obs_dim. v5 changed dimensions for 5 envs (Ant 27→105 being the worst).
**Fix**: Read obs_dim from the live env. Don't store it in config files.
**Affects**: Ant, Humanoid, HumanoidStandup, InvertedDoublePendulum, Reacher.

### 7. Pusher-v4 is broken
**Problem**: `gym.make("Pusher-v4")` raises an error with mujoco≥3.
**Fix**: Use Pusher-v5 only.
**Affects**: Pusher environment.

### 8. `env.reward_range` hidden by wrappers
**Problem**: `env.reward_range` returns None or AttributeError for wrapped envs (TimeLimit, OrderEnforcing).
**Fix**: Access via `env.unwrapped.reward_range`.
**Affects**: Any wrapped env.

### 9. Forgetting `frameskip=1` with AtariPreprocessing
**Problem**: Atari envs default to built-in frame skip. If you also apply AtariPreprocessing with frame_skip=4, you get 16× frame skip (4×4=16).
**Fix**: Pass `frameskip=1` to `gym.make()` when using AtariPreprocessing.
**Affects**: All Atari pixel training.

### 10. Wrong seed placement
**Problem**: `gym.make("Env-v1", seed=42)` — seed is not a make() parameter for most envs.
**Fix**: `env.reset(seed=42)` — seed on reset, not on make.
**Affects**: All envs.

## Medium Impact

### 11. CartPole v0 vs v1
**Problem**: v0 has max_episode_steps=200, v1 has 500. Using v0 makes "solved at 195" impossible.
**Fix**: Use CartPole-v1.
**Affects**: CartPole.

### 12. MountainCarContinuous different reward from MountainCar
**Problem**: Discrete MountainCar gives -1/step. Continuous gives 100 - action_penalty for reaching the goal. Algorithm comparison is meaningless across them.
**Fix**: Compare algorithms within the same env variant.
**Affects**: MountainCar.

### 13. RecordVideo with wrong render_mode
**Problem**: RecordVideo needs `render_mode="rgb_array"` on the base env. Default render_mode is None.
**Fix**: `gym.make(env_id, render_mode="rgb_array")`.
**Affects**: Any env you want to record.

### 14. SyncVectorEnv with envs that share global state
**Problem**: Some envs (Box2D, some MuJoCo versions) have global state. SyncVectorEnv runs them in the same process → state leaks.
**Fix**: Use AsyncVectorEnv for envs with global state, or use different process seeds.
**Affects**: Box2D primarily. MuJoCo v5 is fine with SyncVectorEnv.
