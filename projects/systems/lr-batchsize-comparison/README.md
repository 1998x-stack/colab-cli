# LR x Batch Size Interaction Experiment

Systematic comparison of learning rate and batch size effects on CNN training.
12 experiments across 3 Colab T4 GPUs.

## Experiment Matrix

| | LR=1e-4 | LR=1e-3 | LR=1e-2 | LR=1e-1 |
|---|---|---|---|---|
| **BS=16** | colab | colab | colab | colab |
| **BS=64** | cc | cc | cc | cc |
| **BS=256** | clb | clb | clb | clb |

## Fixed Configuration

- Model: SmallCNN (3 conv + 1 fc, ~94K params)
- Dataset: CIFAR-10 (50K train / 10K test)
- Steps: 4000 optimizer updates per experiment
- LR schedule: Constant (no decay)
- Optimizer: AdamW (wd=0.01, eps=1e-4)
- Precision: AMP FP16

## Files

| File | Purpose |
|------|---------|
| `train.py` | Single experiment runner (`--bs`, `--lr`) |
| `launch.py` | Bootstrap + sequential dispatcher (reads `BS` env var) |
| `fetch.py` | Tar outputs on VM for cron download |
| `watchdog.py` | WebSocket relay keepalive (7-min window) |
| `analyze.py` | Local: merge CSVs, generate comparison plots |

## Execution

See design spec: `docs/superpowers/specs/2026-06-14-lr-batchsize-experiment-design.md`
