# Vision Transformer (ViT) on CIFAR-10

A from-scratch Vision Transformer (patch embedding + TransformerEncoder) trained on CIFAR-10 with 3 experiments: baseline, deeper, and small-patch configurations. Optimized for Kaggle (P100 GPU).

## Usage

```bash
# Local or Kaggle training
python train.py

# Runs 3 experiments sequentially and produces comparison charts.
```

## Model

- **Architecture**: Custom ViT with patch embedding via Conv2d, CLS token, learned positional embeddings, TransformerEncoder (norm_first, GELU, dropout=0.1), LayerNorm, linear classification head.
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.05) with CosineAnnealingLR
- **Precision**: Automatic mixed precision (AMP) via `torch.amp.autocast` + GradScaler
- **Data**: torchvision CIFAR-10 with RandomCrop(32, padding=4) + RandomHorizontalFlip + normalization

### Experiment configurations

| Config | Patch | Depth | Heads | Dim | Params |
|--------|-------|-------|-------|-----|--------|
| vit-baseline | 4 | 6 | 8 | 256 | 4.8M |
| vit-deeper | 4 | 10 | 8 | 256 | 7.9M |
| vit-smallpatch | 2 | 4 | 6 | 192 | 1.8M |

## Key results

| Experiment | Best test acc | Params | Training time |
|------------|--------------|--------|---------------|
| vit-baseline | 71.13% | 4.8M | 479s |
| vit-deeper | 71.83% | 7.9M | 786s |
| vit-smallpatch | 69.72% | 1.8M | 973s |
| **Total** | | | **2240s (37 min)** |

Deeper (10-layer) gave best accuracy but at 1.6x the training time. Small-patch (2x2) underperformed and was slowest due to longer sequence length.

## Gotchas

- P100 GPU (sm_60) requires PyTorch/CUDA reinstall — handled automatically in train.py.
- `norm_first=True` in TransformerEncoderLayer triggers a harmless UserWarning.
- Output structured per-experiment: `{slug}/metrics.jsonl`, `charts.png`, `best_model.pt` + `comparison.png`.
