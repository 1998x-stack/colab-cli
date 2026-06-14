# PPO HalfCheetah-v5: Diagnosis

## Primary Cause: 1 Vectorized Environment

**PPO needs diverse rollouts.** With `num_envs=1`, every update sees one trajectory of 2048 correlated steps. The advantage estimates are noise-dominated, PPO's clipped objective clips everything, and the policy never moves from random.

**Proof from your symptoms:**
- Entropy 8.5 = exactly the theoretical entropy of a 6-dim standard Gaussian — the policy has not updated at all from initialization
- Reward -0.3 = random actions producing no forward movement, just control cost

## Fix

```python
envs = gym.vector.SyncVectorEnv([
    lambda i=i: make_env(i) for i in range(4)
])  # minimum 4, 8 is better

# num_steps * num_envs = updates per iteration
# 512 steps * 4 envs = 2048 observations per update (same total, more diverse)
```

## Secondary Issues

### Missing wrappers
```python
def make_env(rank):
    def _init():
        env = gym.make("HalfCheetah-v5")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
        env.reset(seed=42 + rank)
        return env
    return _init
```

MuJoCo obs are unbounded Box(-inf, inf). Without NormalizeObservation, the value network receives wildly different scales, converges poorly, and produces wrong advantage estimates.

### Learning rate
1e-4 is low for MuJoCo PPO. CleanRL defaults to 3e-4. Combined with the single-env problem, 1e-4 gives vanishingly small updates.

## Summary

Fix priority:
1. **4+ vectorized envs** (SyncVectorEnv) — this is the root cause
2. **Add wrapper stack** (ClipAction → NormalizeObservation → TransformObservation)
3. **Increase lr to 3e-4**
4. Consider reducing num_steps to 512 with more envs for more frequent, diverse updates
