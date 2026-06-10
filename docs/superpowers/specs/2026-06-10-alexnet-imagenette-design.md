# AlexNet Faithful Reproduction on Imagenette — Design Spec

**Date:** 2026-06-10
**Status:** designing

## Goals

1. Faithfully reproduce AlexNet (Krizhevsky et al., NeurIPS 2012) architecture and experimental methodology
2. Run 4 ablation experiments on Imagenette (10-class ImageNet subset) in a 1-hour Colab free-tier T4 session
3. Reproduce paper-equivalent charts: error curves, ablation comparison, conv filter visualization, confusion matrix
4. Set up dual-layer monitoring (local cron + VM watchdog) with FLOPS-based ETA tracking
5. Stream checkpoints every epoch to survive VM death

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | Imagenette 160px (HuggingFace `frgfm/imagenette`) | 10 classes, fits 4 experiments in ~42 min |
| Input resolution | Native 160×160, 128×128 random crop | Clean signal, no upsampling artifacts |
| Architecture | Exact AlexNet (last FC → 10) | Paper-faithful layer dimensions |
| LRN | Omitted | Obsolete since BatchNorm; baseline excludes it |
| GPU split | Single GPU, fused | T4 16GB handles full network; paper split was VRAM hack |
| Validation | 80/20 random split from training set | Paper LR scheduling pattern |
| Eval protocol | 10-view test (4 corners + center, each flipped) | Matches paper evaluation exactly |
| Error metric | Top-1 and Top-3 (not Top-5) | Top-5 trivial on 10 classes |
| Report format | Error rate (%) + accuracy (%) + loss | Error rate for paper charts; all 3 in metrics.json |
| Weight init | Paper init (N(0,0.01) conv; N(0,0.005) FC; bias=1 conv2/4/5+FCs) | Paper explicitly specifies these values |
| Color augmentation | Full PCA color augmentation (Fancy PCA) | Paper-distinctive feature; compute eig from train set |
| Data source | HuggingFace `frgfm/imagenette` with HF_TOKEN auth | Consistent with codebase conventions |
| FC6 dim adaptation | `nn.AdaptiveAvgPool2d(6)` after Conv5 | Matches paper's 6×6 output regardless of input size |
| Account | `colab` (hackxie1998@gmail.com) + `cc` (xbetterdetermine@gmail.com) | Two free GPU slots; run experiments in parallel |
| Session names | `alexnet-a` (colab), `alexnet-b` (cc) | One per account, experiments split across |
| Project dir | `projects/alexnet_imagenette/` | Follows repo conventions (underscore for Python import) |

## Architecture

```
Input: 3 × 160 × 160 (→ 128×128 random crop)
Conv1: 96 × 11×11, stride 4, pad 2 → ReLU → MaxPool 3×3 stride 2
Conv2: 256 × 5×5, pad 2 → ReLU → MaxPool 3×3 stride 2
Conv3: 384 × 3×3, pad 1 → ReLU
Conv4: 384 × 3×3, pad 1 → ReLU
Conv5: 256 × 3×3, pad 1 → ReLU → MaxPool 3×3 stride 2
AdaptiveAvgPool2d(6) → Flatten → 256*6*6 = 9216
FC6: 9216 → 4096, ReLU, Dropout 0.5
FC7: 4096 → 4096, ReLU, Dropout 0.5
FC8: 4096 → 10
```

Parameters: ~57M

## 4 Experiments (split across 2 accounts)

**`colab` → `alexnet-a`** (runs exps 1+2):

| # | Name | Diff from baseline | Time |
|---|---|---|---|
| 1 | Baseline | Full AlexNet as above | ~8 min |
| 2 | No Dropout | Dropout 0 → 0.0 in FC6, FC7 | ~8 min |

**`cc` → `alexnet-b`** (runs exps 3+4):

| # | Name | Diff from baseline | Time |
|---|---|---|---|
| 3 | No Data Aug | No PCA color aug, no random crop, no flip | ~8 min |
| 4 | Reduced Width | Filters halved: 48,128,192,192,128 | ~8 min |

Per-account: ~16 min training. Chart generation + downloads: ~5 min. Session budget: 60 min (each, in parallel). Total wall clock: ~25 min.

## Training Hyperparameters

| Parameter | Value (matches paper) |
|---|---|
| Optimizer | SGD, momentum 0.9 |
| Weight decay | 0.0005 |
| Initial LR | 0.01 |
| LR schedule | ÷10 on validation loss plateau (patience=3) |
| Batch size | 128 |
| Epochs | 90 |
| Loss | Cross-entropy |
| Data augmentation (baseline) | Random 128×128 crop, H-flip, PCA color jitter |

## Monitoring (Dual-Layer)

### Layer 1: Local Cron (every 5 min)

**Two cron jobs**, one per account:

| Cron | Command | Monitors |
|---|---|---|
| `alexnet-a-check` | `colab exec -s alexnet-a -f check_progress.py --timeout 15` | Exps 1+2 |
| `alexnet-b-check` | `cc exec -s alexnet-b -f check_progress.py --timeout 15` | Exps 3+4 |

Each checks:
- VM alive (`colab status`)
- Training process alive (`pgrep python`)
- Last 15 log lines tailed
- Loss health (alert if >5× initial)
- LR drops (flag epoch when lr changes)
- ETA from FLOPS consumed vs expected
- Time remaining (trigger emergency download at <10 min)

### Layer 2: VM Watchdog (background process)

Co-process on VM that:
- Writes heartbeat file every 30s to `/content/heartbeat.json` (tracks: latest epoch, loss, elapsed time, FLOPS consumed)
- On each epoch completion: saves checkpoint `.pt` + appends metrics line to `/content/metrics.jsonl`
- Local cron verifies heartbeat freshness via `cc exec -f check_progress.py` (reads heartbeat timestamp)
- If heartbeat stale >2 min → `check_progress.py` also tries `pgrep python` to distinguish VM-dead from training-stall

### FLOPS Tracking

Values are approximate, scaled from paper's 224×224 to our 128×128 input (FC-dominated portion is unchanged).

| Metric | Value |
|---|---|
| Per-image forward (FP32, 128×128) | ~1.0 GFLOPs |
| Per-image training (fwd+bwd+update) | ~3.0 GFLOPs |
| Imagenette train set | ~7.6K images/epoch (80% of ~9.5K) |
| Per-epoch compute | ~23 TFLOPs |
| T4 effective throughput (~50% util) | ~4 TFLOPS |
| Per-epoch wall time | ~6s |
| 90-epoch experiment | ~9 min (with overhead: ~8 min — Python/pip/transforms dominate on small convs) |

## Output Artifacts

1. **training_curves.png** — Top-1/Top-3 error vs epochs, all 4 experiments overlaid (matching paper Figure 3 style)
2. **ablation_bars.png** — Final test error bar chart, 4 experiments side by side
3. **conv1_filters.png** — 96 Conv1 filters visualized as RGB patches (matching paper Figure 4, left panel)
4. **confusion_matrix.png** — Per-class error pattern for baseline model
5. **metrics.json** — All raw data: per-epoch loss/error, test metrics, FLOPS, timing

## Files

```
projects/alexnet_imagenette/
├── alexnet.py              # Model definition (exact paper architecture)
├── train.py                # Training loop, 4-experiment orchestrator, chart generation
├── launch.py               # Colab bootstrap: pip install, HF token, spawn train + watchdog
├── check_progress.py       # Local cron target: reads heartbeat, tails log, checks health
└── watchdog.py             # VM-side watchdog: heartbeat file writer
```

## Job Flow

```bash
# ═══ Setup: both accounts in parallel ═══

# 1a. Provision session A (colab account, exps 1+2)
colab new --gpu T4 -s alexnet-a

# 1b. Provision session B (cc account, exps 3+4)
cc new --gpu T4 -s alexnet-b

# 2a. Upload to session A
colab upload projects/alexnet_imagenette/launch.py /content/launch.py
colab upload projects/alexnet_imagenette/train.py /content/train.py
colab upload projects/alexnet_imagenette/alexnet.py /content/alexnet.py
colab upload projects/alexnet_imagenette/watchdog.py /content/watchdog.py
colab upload .huggingface/access_token /content/hf_token

# 2b. Upload to session B
cc upload projects/alexnet_imagenette/launch.py /content/launch.py
cc upload projects/alexnet_imagenette/train.py /content/train.py
cc upload projects/alexnet_imagenette/alexnet.py /content/alexnet.py
cc upload projects/alexnet_imagenette/watchdog.py /content/watchdog.py
cc upload .huggingface/access_token /content/hf_token

# 3a. Launch session A (exps 1,2) — exp_ids come from uploaded exp_ids.txt
colab exec -s alexnet-a -f launch.py --timeout 120

# 3b. Launch session B (exps 3,4)
cc exec -s alexnet-b -f launch.py --timeout 120

# ═══ Monitoring ═══
# CronCreate: 2 jobs, each every 5 min
#   Job A: colab exec -s alexnet-a -f check_progress.py --timeout 15
#   Job B: cc exec -s alexnet-b -f check_progress.py --timeout 15

# ═══ Download ═══
# Session A results
colab download /content/checkpoints.tar.gz ./projects/alexnet_imagenette/output-a/
colab download /content/*.png ./projects/alexnet_imagenette/output-a/

# Session B results
cc download /content/checkpoints.tar.gz ./projects/alexnet_imagenette/output-b/
cc download /content/*.png ./projects/alexnet_imagenette/output-b/

# ═══ Cleanup ═══
colab stop -s alexnet-a
cc stop -s alexnet-b
```

## Success Criteria

- [ ] Baseline reaches >60% top-1 accuracy on Imagenette (random chance = 10%)
- [ ] Dropout ablation shows validation error gap vs baseline >10 percentage points
- [ ] No Data Aug ablation shows largest test error increase
- [ ] Reduced Width drops test accuracy measurably
- [ ] All 4 png artifacts render correctly
- [ ] metrics.json has complete per-epoch data for all experiments
- [ ] VM survives full ~45 min run (cron confirms no mid-session death)
- [ ] FLOPS estimates within 30% of actual wall time
