# Knowledge Distillation: ResNet Teacher → Tiny Student on CIFAR-10

Design spec for `projects/cv/knowledge-distillation/`. June 14, 2026.

## Overview

Classic Hinton knowledge distillation: a large pre-trained teacher (ResNet-18, 11M params) transfers knowledge to an aggressively small student (TinyResNet, ~246K params) via softened output distributions. Deployed on Colab T4 GPU with 4-experiment ablation (no-KD + 3 temperatures), relay handoff for session survival, cron watchtower for remote monitoring.

## Goals

- Demonstrate extreme compression: student ≤ 0.5M params achieving meaningful accuracy via distillation
- Run systematic ablation: 4 experiments comparing no-KD baseline vs KD at T ∈ {2, 4, 8}
- Deploy on free-tier Colab T4 with relay handoff for the ~12-min session
- Produce structured outputs for glance-and-decide monitoring via cron

## Architecture

### Teacher

`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` — pre-trained, frozen, eval mode. CIFAR-10 images upsampled to 224×224 via `F.interpolate`. Teacher test accuracy ~94.3% on CIFAR-10.

### Student: TinyResNet

Pure feedforward CNN, no residual connections:

| Layer | Input → Output | Kernel | Params |
|-------|---------------|--------|--------|
| Conv1 | 3→16 | 3×3, pad=1 | 432 |
| BN + ReLU | | | |
| Conv2 | 16→32 | 3×3, stride=2, pad=1 | 4,608 |
| BN + ReLU | | | |
| Conv3 | 32→64 | 3×3, stride=2, pad=1 | 18,432 |
| BN + ReLU | | | |
| Conv4 | 64→128 | 3×3, stride=2, pad=1 | 73,728 |
| BN + ReLU | | | |
| Conv5 | 128→128 | 3×3, pad=1 | 147,456 |
| BN + ReLU | | | |
| AdaptiveAvgPool | 128→128 (1×1) | | |
| FC | 128→10 | | 1,290 |

**Total: ~246K params** (2.3% of ResNet-18). Batch norm after every conv. No dropout. Input kept at native 32×32.

Design rationale: progressive channel doubling gives capacity where feature maps are smallest. Three stride-2 downsampling stages take 32×32 → 4×4. No residuals — the student is a different architecture family from the teacher, making this a true transfer demonstration.

## Loss Design

**KD loss (Hinton 2015):**

```
L_KD = T² × KL( softmax(z_teacher / T) || softmax(z_student / T) )
```

T² scaling compensates for gradient magnitude reduction from the temperature. Pure KD — no hard-label cross-entropy component. Single hyperparameter T.

**No-KD baseline:** standard cross-entropy with hard labels.

## Experiments

Four experiments, one script gated by `--exp_id`:

| exp_id | Mode | T | Purpose |
|--------|------|---|---------|
| `a` | no-KD baseline | — | Lower bound: student from scratch |
| `b` | KD | 2 | Sharp-ish soft targets |
| `c` | KD | 4 | Classic Hinton default |
| `d` | KD | 8 | Softer, reveal inter-class structure |

**Training budget per experiment:** 30 epochs, batch 128, AdamW(lr=1e-3, weight_decay=1e-4), cosine LR to 1e-5. Same student init seed per experiment for fair comparison. ~2 min each on T4. Total ~12 min including data prep and teacher loading.

## Files

```
projects/cv/knowledge-distillation/
├── train.py              # Student model + training loop + 4-experiment dispatch
├── launch.py             # Bootstrap: pip install, spawn detached train.py
├── watchdog.py           # Relay handoff: heartbeat + sentinel polling
├── check_progress.py     # Quick status: process alive, log tail, checkpoint check
├── fetch.sh              # Cron: tar on VM, download, extract, print tail + comparison
├── exp_ids.txt           # a\nb\nc\nd — experiment list for launch.py
├── gotchas.md            # Project-specific gotchas
└── README.md             # Results summary
```

### Shared utilities (copied from `.claude/skills/colab-cli/scripts/`)
- `log_utils.py` — Logger, Tee, MetricsCSV, SummaryJSON, detect_output_dir
- `plot_utils.py` — plot_loss_acc (per-experiment), custom comparison figure

## Output Structure

```
/content/kd-output/
├── logs/train.log            # Unified log, per-batch lines prefixed with [exp_<id>]
├── metrics.csv               # All experiments, exp_id column for grouping
├── pngs/
│   ├── training_curves.png   # Overwritten per experiment (4-panel loss/acc/LR/dist)
│   └── comparison.png        # Final 4-experiment comparison grid
├── checkpoints/exp_<id>_best.pt  # Best student weights per experiment
└── summary.json              # Final comparison table
```

## Log Format

Per-batch (self-contained, copy-paste-comparable):

```
[HH:MM:SS] [exp_a] Epoch 15/30 | Batch 300 | loss=0.8472 | avg100=0.8531 | acc=0.46 | lr=0.000832 | elapsed=98s
[HH:MM:SS] [exp_c] Epoch 15/30 | Batch 300 | kd_loss=2.3412 | avg100=2.3510 | train_acc=0.52 | T=4 | lr=0.000832 | elapsed=102s
```

Epoch-end:

```
[HH:MM:SS] === [exp_c] Epoch 30/30 done | train_acc=0.78 | test_acc=0.7612 | T=4 | time=128s ===
```

## Metrics CSV

```
exp_id,epoch,train_loss,train_acc,test_acc,temperature,elapsed_s,lr
a,1,0.9234,0.450,0.501,,62.3,0.001000
b,1,3.1234,0.420,0.482,2,63.1,0.001000
c,30,2.0891,0.780,0.761,4,128.5,0.000010
```

## Visualization

**comparison.png** (custom 4-panel):
1. Test accuracy vs. epoch — 4 lines + teacher horizontal reference
2. Training loss curves — CE for no-KD, KD loss for rest
3. Bar chart — final test acc with teacher baseline
4. Table summary — exp_id, T, test_acc, params, distillation gap

**training_curves.png** (reuses `plot_loss_acc`, overwritten per experiment):
Loss curve + eval accuracy + LR schedule + loss distribution histogram.

## summary.json

```json
{
  "teacher": {"model": "resnet18", "params": 11176522, "test_acc": 0.943},
  "student_arch": {"name": "tinyresnet", "params": 246342},
  "results": [
    {"exp_id": "a", "mode": "no_kd", "T": null, "test_acc": 0.731, "time_s": 121},
    {"exp_id": "b", "mode": "kd", "T": 2, "test_acc": 0.752, "time_s": 128},
    {"exp_id": "c", "mode": "kd", "T": 4, "test_acc": 0.761, "time_s": 128},
    {"exp_id": "d", "mode": "kd", "T": 8, "test_acc": 0.769, "time_s": 130}
  ],
  "best": {"exp_id": "d", "test_acc": 0.769, "distillation_gap": 0.174},
  "total_time_s": 507
}
```

## Colab Deployment

### Proxy
Config B (HTTP CONNECT tunnel) for full-session reliability:
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

### Session flow
```bash
colab new --gpu T4 -s kd-cifar10
colab upload train.py launch.py watchdog.py /content/
colab exec -f launch.py                    # pip install + spawn detached train.py
nohup colab exec -f watchdog.py --timeout 420 &  # redundant ×2, 30s apart
```

### Relay handoff
Two redundant watchdogs, 6-min window, launched 30s apart. Each runs a heartbeat (print timestamp + `nvidia-smi | head` every 25s) while polling `/content/kd-output/DONE`. `train.py` writes `DONE` on completion. Watchdog exits on sight. Training process is detached (`start_new_session=True`) — survives all WebSocket state.

### Cron watchtower
`fetch.sh` fires every 2 min via `CronCreate`:
1. Check session alive: `colab sessions | grep kd-cifar10`
2. Tar on VM: `echo 'subprocess.run(["tar", "-czf", ...])' | colab exec`
3. Download tar → extract locally
4. Print log tail (last 3 lines) + metrics.csv last row per experiment + comparison

Cron prompt:
```
Check session alive (colab sessions | grep kd-cifar10). Tar kd-output on VM. Download tar to projects/cv/knowledge-distillation/output/. Extract. Print tail -5 of logs/train.log. Print last row per exp_id from metrics.csv. If session dead, report FATAL.
```

## Pre-deploy Checklist

- [ ] `ruff check .` passes with zero errors
- [ ] Forward pass: random (4, 3, 32, 32) → teacher (upsampled 224×224) and student produce correct output shapes
- [ ] Teacher logits shape = (4, 10), student logits shape = (4, 10)
- [ ] KD loss is scalar and finite
- [ ] Time budget: ~12 min (3 min data+load + 4×2 min experiments + 1 min summary). Relay handoff required.

## Edge Cases

- **Teacher returns NaN logits:** torchvision pre-trained weights are stable on CIFAR-10. Validate first batch.
- **Student loss diverges:** KD loss with T² scaling should be stable. If not, lower LR to 5e-4.
- **Session dies mid-experiment:** Checkpoints saved per epoch. Re-launch with remaining exp_ids.
- **exp_id collision in metrics:** CSV uses exp_id column. Each experiment appends its own rows. No collision possible.
- **NAT timeout during relay:** 25s heartbeat with `nvidia-smi` produces real TCP payload. Two redundant watchdogs give ~84% chance of at least one surviving each handoff.

## What This Is NOT

- Not a general distillation framework — specific to CIFAR-10 + ResNet teacher
- Not feature-based distillation (FitNets, AT, RKD) — pure response-based Hinton KD
- Not for production deployment — educational project demonstrating the core concept
- Not multi-teacher or self-distillation
