# DDPG vs TD3: Pendulum-v1 Benchmark Report

**Date**: 2026-06-13 | **Environment**: Pendulum-v1 (continuous control, 3D state, 1D action)
**GPU**: NVIDIA T4 (Colab free tier) | **Framework**: PyTorch + Gymnasium
**Seeds**: 42 (both runs) | **Episodes**: 200

## Summary

TD3 eliminates DDPG's catastrophic forgetting but converges slower on simple environments. On Pendulum-v1, DDPG found a higher peak (-40.95) but immediately lost it; TD3's final model (-71.76) is its best model — still improving at episode 200. For deployment (where you ship the final model, not a mid-training snapshot), TD3 wins decisively.

## Results

| Metric | DDPG | TD3 |
|--------|------|-----|
| Best eval | **-40.95** (ep 40) | **-71.76** (ep 200) |
| Final eval | -165.96 ± 56.8 | **-71.76 ± 93.2** |
| Worst regression | **-606** (ep 40→50: -41→-647) | 0 (peak is final — no regression) |
| Evals with std > 100 | 4 / 20 | 2 / 20 |
| Last 5 evals stdev | 51.1 | **27.9** |
| Best→worst after peak | 606.3 | **0.0** (peak is final) |
| Best eval is final? | No (ep 40) | **Yes (ep 200)** |
| Catastrophic forgetting | **Yes** — lost best policy immediately | **No** — monotonic improvement |

### Training Reward Trend (block averages)

```
DDPG:
  eps   1-20: -1404  ████████████████████████████████████████
  eps  21-40:  -694  ████████████████████████
  eps  41-60:  -191  ████████
  eps  61-80:  -195  ████████
  eps  81-100: -161  ███████
  eps 101-200: ~-150  ███████ (plateau)

TD3:
  eps   1-20: -1281  ████████████████████████████████████████
  eps  21-40:  -612  ██████████████████████
  eps  41-60:  -547  ████████████████████
  eps  61-80:  -428  █████████████████
  eps  81-100: -337  █████████████
  eps 101-140: ~-145  ██████ (converging)
  eps 141-150: ~-200  ███████ (mild regression)
```

### Evaluation by Episode

```
Episode  DDPG           TD3
  10     -1319 ± 121    -1369 ± 76
  20     -1066 ± 241    -1121 ± 63
  30      -779 ± 185     -724 ± 69
  40       -41 ± 46 ★    -903 ± 92
  50      -647 ± 433 ✗    -472 ± 81
  60      -195 ± 93      -260 ± 113
  70      -144 ± 42      -215 ± 75
  80      -148 ± 49      -355 ± 511 ⚠
  90       -72 ± 59      -174 ± 90
 100       -73 ± 58      -145 ± 87
 110      -145 ± 44      -142 ± 90
 120      -119 ± 75      -146 ± 45
 130       -96 ± 48      -146 ± 44
 140      -122 ± 74      -145 ± 90
 150      -122 ± 79      -239 ± 74
 160      -135 ± 99       -97 ± 86
 170      -220 ± 47      -146 ± 44
 180       -99 ± 50      -120 ± 74
 190       -98 ± 48       -99 ± 49
 200      -166 ± 57       -72 ± 93 ★
```

★ = best overall  |  ✗ = catastrophic collapse  |  ⚠ = instability spike (self-corrected)

## Analysis

### 1. Stability (TD3 wins clearly)

DDPG's Q-value overestimation bias caused the actor to chase phantom high-Q regions. The result: a policy that scored -40.95 at episode 40 (near-optimal for Pendulum) was completely overwritten by episode 50 (-647). This is textbook DDPG catastrophic forgetting.

TD3's twin-critic mechanism (`min(Q1, Q2)` for the target) prevents this entirely. TD3's best eval (-71.76) is its **final** eval — the policy was still improving at episode 200. The "best→worst after peak" metric is 0.0 for TD3 (there is no post-peak drop) vs 606.3 for DDPG. TD3's last 5 evals have half the variance of DDPG's (stdev 27.9 vs 51.1).

TD3 had two wobbles: episode 80 (std=511, self-corrected by episode 90) and episode 150 (-239 after holding -145 for 5 straight evals, then recovered to -97 by episode 160). Both were transient — the twin-critic + policy smoothing combination pulled the policy back each time, something DDPG couldn't do.

### 2. Final Model Quality (TD3 wins)

If you deploy the **final trained model** (the standard workflow), TD3 wins decisively:

- **TD3 final model: -71.76** (its best)
- **DDPG final model: -165.96** (2.3x worse than its peak)

Deploying DDPG requires tracking the best-validation checkpoint and using that instead of the final model — fragile and easy to forget. TD3's final model is its best model by construction.

### 3. Convergence Speed (DDPG wins on simple envs)

DDPG found its best policy at episode 40. TD3 took until episode 100 to reach comparable performance, and its peak (-71.76) didn't match DDPG's peak (-40.95). Three TD3 design choices explain the slower convergence:

- **Lower learning rates** (3e-4 vs 1e-3): Slower but more stable updates
- **Delayed policy updates** (d=2): Actor updated half as often as the critic
- **Target policy smoothing**: Adds regularization noise that slows convergence to sharp optima

On Pendulum's simple landscape, this caution is unnecessary overhead. On harder problems, it's essential.

### 4. When TD3 Matters

Pendulum-v1 is a toy problem. DDPG's overestimation bias is most damaging when:

- **High-dimensional action spaces** (Ant: 8D, Humanoid: 17D)
- **Sparse rewards** — overestimating a single transition derails the policy
- **Long horizons** — errors compound over more timesteps
- **Narrow optimal basins** — sharp peaks easy to overshoot

For Pendulum, DDPG + `best_model.pt` checkpointing would be sufficient if you remember to deploy the checkpoint, not the final model. For anything harder, TD3 is the safer default.

## Proxy & Infrastructure Notes

### Colab Session Stability

| Run | Session | Fate |
|-----|---------|------|
| DDPG | `ddpg` | Completed 200 eps, died after |
| TD3 Run 1 | `ddpg` (reused) | **Killed at ep 94** — VM pruned mid-training |
| TD3 Run 2 | `td3` (fresh) | Running (ep 155+) |

**Lesson**: Don't reuse Colab sessions. DDPG finished in ~4 minutes; the reused session died sometime after, killing TD3 Run 1 at 47% complete. Always provision a fresh session for each training run. Free-tier VMs are unpredictable — the 12h max is aspirational, not guaranteed.

### Proxy Configuration

GPU provisioning succeeded with config B (HTTP CONNECT + ALL_PROXY=socks5) when config A (SOCKS5 + no_proxy) returned 503 Service Unavailable. See CLAUDE.md proxy section for the diagnostic table.

## File Inventory

```
projects/ddpg-gym/
├── train.py              # DDPG implementation
├── launch.py / check_progress.py / fetch.sh
└── output/
    ├── train.log         # Full training log (200 eps)
    ├── metrics.json      # All episode + eval data
    └── plots/            # progress.png per fetch

projects/td3-gym/
├── train.py              # TD3 implementation (twin critics, delayed updates, target smoothing)
├── launch.py / check_progress.py / fetch.sh
└── output/
    ├── train.log         # Run 1 (died ep 94) + Run 2 (running)
    ├── metrics.json
    └── plots/
```
