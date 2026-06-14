# PPO HalfCheetah-v5 Debug

## Diagnosis

**Both symptoms together prove the policy is still random.**

- Entropy 8.5 = 1.42 * 6 (6-dimensional Gaussian with std=1 at init)
- Reward -0.3 = random torques, no forward movement

## Root Causes

### Issue 1: Too few timesteps (primary)
51,200 total steps (25 iter × 2048 steps). HalfCheetah needs 1M+. First positive reward at ~100K-200K.

### Issue 2: Gaussian policy with bounded actions
HalfCheetah actions are Box(-1,1). With std=1.0, ~32% of sampled actions get clipped. The log_std gradient averages to zero — it never shrinks.

### Issue 3: No observation normalization
HalfCheetah has 17 dims with very different scales (positions ~[-1,1], velocities ~[-10,15]). Raw observations confuse the value network.

### Issue 4: Single environment
1 env = correlated trajectory segments = high gradient variance.

### Issue 5: Reward scale
Rewards around -0.3 make tiny advantages, meaning tiny policy gradients.

## Fix

1. Run 1M+ timesteps
2. Use squashed Gaussian (tanh) with log_std init = -1.5
3. Normalize observations with running mean/std
4. Use 8 vectorized envs
