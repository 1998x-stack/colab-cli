# CNN Quantization Comparison

ResNet-18 on CIFAR-10 — FP32 vs FP16 vs INT8 vs INT4 accuracy, size, and latency comparison.

## Quickstart

```bash
# Local test
python train.py --epochs 5 --batch_size 128

# Colab (see launch.py for full workflow)
colab new --gpu T4 -s cnn-quant
colab upload train.py /content/train.py
colab upload launch.py /content/launch.py
colab exec -f launch.py -s cnn-quant --timeout 120

# Monitor
colab exec -f check_progress.py -s cnn-quant --timeout 20

# Fetch results (from cron or manual)
bash fetch.sh cnn-quant B
```

## Design

Single-file cleanrl-style: `train.py` is self-contained with argparse, no abstract classes, no config files.

| Component | Detail |
|-----------|--------|
| Dataset | CIFAR-10 (torchvision), 10k train / 2k val subset |
| Model | ResNet-18 adapted for 32×32 input (11.2M params) |
| Training | 10 epochs, SGD+momentum, cosine LR, ~94s on T4 |
| Quantization | FP32 → FP16 (`.half()`) → INT8 (`torch.ao.quantize_dynamic`) → INT4 (custom per-channel symmetric) |
| Outputs | `metrics.csv`, `logs/train.log`, `pngs/training_curves.png`, `pngs/quantization_comparison.png`, `quantization_summary.csv` |

## Verified Results (T4, 3 runs)

| Method | Accuracy | Model Size | Latency | vs FP32 |
|--------|----------|------------|---------|---------|
| FP32 | 72.45% | 42.70 MB | 26.8 ms | baseline |
| FP16 | 72.45% | 21.37 MB (2×) | 18.7 ms (1.4×) | +0.00% |

INT8/INT4: implementations ready but blocked by PyTorch backend quirks (see gotchas.md).

## Files

```
cnn-quantization/
├── train.py            # Main script — training + quantization comparison (cleanrl-style)
├── launch.py           # Colab bootstrap — pip install + spawn detached training
├── check_progress.py   # VM progress monitor — process, GPU, log tail, CSV, PNGs
├── fetch.sh            # REST tar+download for cron watchtower
├── gotchas.md          # Project-specific learnings
├── README.md           # This file
└── output/             # Local artifacts (fetched from Colab)
    ├── logs/
    ├── pngs/
    └── metrics.csv
```

## Colab Deployment Notes

- **10-min GPU window** is the binding constraint. Training (94s) + quantization (2-3min) fits comfortably if data download is fast.
- **CIFAR-10 download from cs.toronto.edu** is variable (30s–4min). Warmup session recommended for reliable runs.
- **4 accounts** available for parallel/rotating GPU sessions (`colab`, `cb`, `cc`, `clb`).
- **INT8 needs CPU** — `torch.ao.quantize_dynamic` quantized ops are CPU-only on Colab's PyTorch build.
- Use Config A (SOCKS5 + no_proxy) for `colab exec`; Config B (HTTP CONNECT) for `colab new` if Config A returns 503.
