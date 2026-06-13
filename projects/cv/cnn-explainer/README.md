# CNN Explainer

A CNN classifier on CIFAR-10 with explainability visualizations: Grad-CAM, Saliency Maps, Integrated Gradients, Feature Map activation analysis, and sample prediction grids.

## Usage

```bash
# Local training (with optional args)
python train.py [--epochs 10] [--batch-size 128] [--lr 0.001]

# Colab deployment
cb launch.py
```

## Model

- **Architecture**: 3x ConvBlock(3→32, 32→64, 64→128) + AdaptiveAvgPool2d + Linear(128→10)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR
- **Training**: 10 epochs (default), gradient clipping at 1.0
- **Data**: 80/20 train/val split from HuggingFace `uoft-cs/cifar10`

### Explainability techniques

| Technique | Description |
|-----------|-------------|
| Grad-CAM | Class activation heatmap from last conv layer |
| Saliency Map | Gradient magnitude (max across RGB channels) |
| Integrated Gradients | Path integral of gradients, 20 steps, black baseline |
| Feature Maps | Top-1 activating image per random filter from each conv block |

## Key results

| Metric | Value |
|--------|-------|
| Validation accuracy (best) | 68.82% |
| Test loss | 0.893 (final epoch) |
| Training time | ~6 min (10 epochs) |
| Model parameters | ~145K |

## Gotchas

- Grad-CAM uses `register_forward_hook` + `register_full_backward_hook` on `model.last_conv`.
- Saliency uses max across RGB channels (not sum or L2 norm).
- Integrated Gradients uses a black (zero) baseline.
- `weights_only=True` in `torch.load` per PyTorch 2.6+ security defaults.
- If CUDA OOM, reduce `--num-explain` from default 16 to 8.
