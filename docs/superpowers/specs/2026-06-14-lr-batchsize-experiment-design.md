# LR × Batch Size Interaction Experiment

Date: 2026-06-14 | Colab T4 | PyTorch 2.11

## Goal

Systematically measure how learning rate and batch size interact during CNN training. Verify the linear scaling rule (double BS → double optimal LR), find where it breaks down, and produce a 2D heatmap of convergence quality.

## Experiment Matrix

12 experiments: 3 batch sizes × 4 learning rates.

| | LR=1e-4 | LR=1e-3 | LR=1e-2 | LR=1e-1 |
|---|---|---|---|---|
| **BS=16** | colab | colab | colab | colab |
| **BS=64** | cc | cc | cc | cc |
| **BS=256** | clb | clb | clb | clb |

Reserve accounts: colb, clab (used if any primary account hits GPU exhaustion mid-run).

## Fixed Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | SmallCNN (3 conv + 1 fc, ~1.2M params) | Existing benchmark model, known behavior |
| Dataset | CIFAR-10 (50K/10K) | Standard, fast download, fits in T4 VRAM |
| Steps | 4000 optimizer updates | Equal compute budget per experiment |
| LR schedule | Constant (no decay) | One variable at a time |
| Optimizer | AdamW (β=0.9, 0.999, wd=0.01) | Standard. eps=1e-4 for AMP safety |
| Precision | AMP FP16 (GradScaler) | Unlocks T4 tensor cores |
| Eval frequency | Every 200 batches | 20 eval points per run |
| Seed | 42 | Reproducibility |

## Train Script Design

One script: `train.py`. Takes `--bs` and `--lr` as arguments (e.g., `python train.py --bs 16 --lr 1e-3`). Runs 4000 steps with that config, writes outputs, exits. The launch.py dispatcher calls it 4 times sequentially with the 4 LR values for that account's assigned batch size.

## Per-Experiment Outputs

Each experiment writes to `/content/lr-bs-output/bs<BS>_lr<LR>/` (e.g., `bs16_lr0.001/`):

```
logs/train.log       — timestamped per-batch: loss, acc, grad_norm
metrics.csv          — batch,loss,train_acc,test_acc,lr,grad_norm,elapsed_s
pngs/loss_acc.png    — loss + accuracy curves (overwritten every 500 batches)
summary.json         — config, best_acc, final_acc, total_time_s
```

Log format:
```
[HH:MM:SS] Batch 400 | loss=1.2345 | train_acc=0.456 | lr=0.001000 | grad_norm=2.345
[HH:MM:SS]   -- Eval @ 400: test_loss=1.567 | test_acc=0.423
```

## Comparison Outputs (post-hoc, local)

Generated after all experiments complete:

1. **Heatmap** (`lr_bs_heatmap.png`): LR × BS → final test accuracy. 3×4 colored grid.
2. **Overlay curves** (`lr_curves.png`): 3 panels (one per BS), 4 LR curves overlaid. Loss and accuracy vs batch.
3. **Optimal LR vs BS** (`optimal_lr_vs_bs.png`): Scatter plot — best LR per batch size. Reference line for linear scaling.
4. **Gradient noise** (`gradient_noise.png`): Gradient norm std vs batch size.
5. **Comparison CSV** (`all_experiments.csv`): All 12 runs merged — bs, lr, best_acc, final_acc, steps_to_90pct (steps to reach 90% of best accuracy), grad_norm_std, total_time_s.

## Multi-GPU Execution Plan

### Phase 1: Data warmup (sequential, ~8 min)
Provision each account with T4 GPU, trigger CIFAR-10 download (torchvision caches to `/content/data/`), stop session. This ensures Phase 2 doesn't waste GPU window on the ~170MB download.

```bash
# Per account — inline download trigger (runs in seconds after download completes):
colab new --gpu T4 -s warmup
echo 'import torchvision.datasets; torchvision.datasets.CIFAR10(root="/content/data", train=True, download=True); print("cached")' | colab exec -s warmup --timeout 120
colab stop -s warmup
# Repeat for cc, clb
```

### Phase 2: Parallel training (~30 min wall clock)
All 3 accounts simultaneously. Each account runs 4 experiments (one BS, 4 LRs) sequentially. Estimated 20-28 min per account → requires WebSocket relay handoff.

Launch pattern (per account):
```bash
# Provision
HOME=~/colab-accounts/account-<X> colab new --gpu T4 -s lrbs

# Upload scripts
HOME=... colab upload train.py /content/train.py
HOME=... colab upload launch.py /content/launch.py

# Launch detached
HOME=... colab exec -f launch.py --timeout 120
```

`launch.py` installs matplotlib, spawns `train.py --exp_ids bs16_lr1e-4,bs16_lr1e-3,...` as detached subprocess.

### Phase 3: Cron monitoring (3 parallel crons)
One CronCreate job per account, firing every 3 minutes. Each tick: tar outputs on VM via uploaded `fetch.py`, download via REST, extract, report tail.

```
# Cron prompt template (per account):
# 1. Check session alive: HOME=~/colab-accounts/account-X colab sessions | grep lrbs
# 2. Tar on VM: HOME=... colab exec -s lrbs -f fetch.py --timeout 15
# 3. Download: HOME=... colab download -s lrbs /content/out.tar.gz projects/systems/dl-training/output/<account>_out.tar.gz
# 4. Extract: tar -xzf output/<account>_out.tar.gz -C output/<account>/
# 5. Report: tail -20 output/<account>/<exp_dir>/logs/train.log
```

`fetch.py` is pre-uploaded to the VM at launch time. Uses `colab exec -f` (not stdin pipe) for reliability with multi-line scripts.

### Phase 4: Analysis (local)
Merge CSVs → generate 5 comparison artifacts → write summary.

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU exhaustion mid-run (412) | Move remaining experiments to colb/clab reserve |
| WebSocket relay fails | Cron download still works (REST path). Data not lost. |
| LR=1e-1 diverges immediately | Script detects loss > 3× initial → aborts that run, logs DIVERGED |
| BS=256 OOMs T4 | SmallCNN on CIFAR-10 fits easily. BS=512 would be tight but we cap at 256. |
| First session data download eats GPU window | Phase 1 warmup sessions cache data before Phase 2 |

## Success Criteria

1. All 12 experiments complete within 2 hours wall clock
2. Heatmap clearly shows optimal (LR, BS) region
3. Linear scaling rule verified or refuted with evidence
4. At least 3 of 5 comparison artifacts generated
5. All outputs committed to repo with analysis summary
