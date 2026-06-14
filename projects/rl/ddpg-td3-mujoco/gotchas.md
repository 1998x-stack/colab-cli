# DDPG vs TD3 MuJoCo — Gotchas

## Kaggle P100: PyTorch CUDA mismatch

P100 (sm_60) is incompatible with PyTorch 2.10+ cu128. The train.py auto-detects and reinstalls torch for CUDA 12.6 before any `import torch`. This takes ~2-3 minutes on first run but prevents silent CPU fallback.

## MuJoCo rendering: MUJOCO_GL=egl required

Without `MUJOCO_GL=egl`, MuJoCo env creation fails on headless VMs (Colab/Kaggle) because there's no display. The launch.py sets this automatically.

## Total runtime: 3-6 hours for full benchmark

6 (env, algo) pairs × ~30-60 min each = 3-6 hours total. This cannot run in a single free-tier Colab session (~8 min GPU window). Use Kaggle (9h session) or run one pair per Colab session.

Run a single pair:
```bash
python -c "
from train import train_one
train_one('DDPG', 'HalfCheetah-v4', 400, '/content/ddpg-td3-mujoco-output/HalfCheetah-v4/DDPG')
"
```

## HalfCheetah solved at ~4000+ reward

HalfCheetah-v4 with DDPG reaches ~4400 eval reward at 360 episodes (best). TD3 typically reaches higher (~6000+) but takes more episodes. Hopper solved at ~2500+. Walker2d solved at ~3000+.

## Gradient clipping (1.0) prevents value explosion

MuJoCo envs have large state/action magnitudes. Without clip_grad_norm_(1.0), the critic can diverge in early training. Applied to both actor and critic for both DDPG and TD3.

## Kaggle vs Colab: CUDA JIT on first run

First run on any platform compiles CUDA kernels for MuJoCo ops. This adds 2-5 min overhead. Second run is fast.

## Memory: 1M replay buffer fits T4 15.6GB

The replay buffer uses CPU RAM (deque), not GPU VRAM. GPU memory holds only batch tensors. With batch_size=256 and max state/action dims (17+6 for HalfCheetah), GPU RAM usage is ~2-3 GB.

## TD3's delayed policy updates matter

TD3 updates the actor every 2 critic updates (`POLICY_DELAY=2`). Reducing this to 1 (same as DDPG) eliminates TD3's advantage — the twin critics need time to converge before the actor uses them.

## No summary.json in nested output

Each (env, algo) pair writes to its own subdirectory. The master comparison plots are at `{out_root}/comparison/`. Individual pair results are at `{env}/{algo}/metrics.csv` and `{env}/{algo}/train.log`.
