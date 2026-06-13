# DDPG vs TD3 on MuJoCo

Head-to-head comparison of DDPG and TD3 on three MuJoCo continuous-control environments (HalfCheetah-v4, Hopper-v4, Walker2d-v4), with per-env comparison plots and a master dashboard.

## Usage

```bash
# Local training (all 3 envs, both algos, sequentially)
python train.py

# Colab deployment
cb launch.py [--args]
```

## Key results

Partial results (training interrupted). Completed run on **DDPG on HalfCheetah-v4** (300/400 episodes):

| Metric | Value |
|--------|-------|
| Best eval mean reward | 4414.93 (episode 360, HalfCheetah-v4, DDPG) |
| Final avg100 reward | ~4200 (HalfCheetah-v4, DDPG) |
| Total training time (300 eps) | ~60 minutes (CUDA) |
| Episodes per env | HalfCheetah: 400, Hopper: 300, Walker2d: 300 |
| Batch size | 256 |
| Replay buffer | 1,000,000 transitions |
| Shared gamma | 0.99, tau = 0.005 |

### DDPG hyperparameters

| Parameter | Value |
|-----------|-------|
| Actor learning rate | 1e-3 |
| Critic learning rate | 1e-3 |
| Exploration | OU noise (theta=0.15, sigma=0.2) |

### TD3 hyperparameters

| Parameter | Value |
|-----------|-------|
| Actor learning rate | 3e-4 |
| Critic learning rate | 3e-4 |
| Policy delay | 2 |
| Policy noise | 0.2 (clipped 0.5) |
| Exploration noise | 0.1 (Gaussian) |

## Gotchas

- On Kaggle P100 GPUs (sm_60), PyTorch 2.10+ with CUDA 12.5/12.8 is incompatible and auto-reinstalled for CUDA 12.6.
- All six (env, algo) combinations run sequentially; total training can take several hours.
- Sets `MUJOCO_GL=egl` for headless rendering on Colab/Kaggle VMs.
- Output organized as `{out_root}/{env_name}/{algo}/` with per-env training curves.
- Comparison plots generated after both algos finish each environment, plus a master 3x2 dashboard.
- Gradient clipping (max norm 1.0) applied to both actor and critic networks.
