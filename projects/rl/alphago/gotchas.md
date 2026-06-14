# AlphaGo Zero — Gotchas

Project-specific issues and lessons learned.

## Eval dominates time budget

Single-tree MCTS at 400 sims for eval is ~10s per game on T4. 100 eval games = ~17 minutes — exceeds the free-tier GPU window. For Colab sessions, use `n_eval_games=20` or lower. Full eval suites should run on Kaggle (P100, 9h window).

## First session always fails training

Colab VM download + CUDA JIT + first-time Gymnasium install = 7-10 min overhead. Combined with ~8 min effective GPU window for training, the first session almost never completes. Use `CONFIG_9X9_FIRST` (20 games, 100 sims, 3 epochs, 1 eval game) as a warmup only.

## NaN/Inf in value head

With small batch sizes (<32) and high LR (>0.01), the value head can diverge to NaN. The train.py NaN guard catches this and saves checkpoint before exit. Recovery: reload checkpoint, halve LR.

Fix applied: `batch_size=64`, `lr=0.001` with SGD+momentum (not Adam — AGZ paper uses SGD).

## Duplicate libomp warning

`KMP_DUPLICATE_LIB_OK=TRUE` is set in train.py. Colab pre-loads multiple OpenMP runtimes which causes a harmless warning. Setting this env var suppresses it.

## Drive mount required for checkpoints

Without Drive mount, checkpoints only exist on VM and vanish with the session. Always verify Drive mount succeeded before training: `ls /content/drive/MyDrive/`.

## MCTS batch runner shares model

The `MCTSBatchRunner` uses a single model instance for all parallel games. If you modify the model during training, the batch runner sees the updated weights. This is correct for AlphaGo Zero (train immediately after self-play) but means you can't run self-play and training concurrently without model copying.

## 19×19 is infeasible on free-tier Colab

`CONFIG_19X19` (20 residual blocks, 256 filters) ≈ 5M params. A single MCTS search at 800 sims takes ~30s. Self-play of 50 games = 25+ minutes just for self-play. Use Kaggle P100 or paid Colab for 19×19.
