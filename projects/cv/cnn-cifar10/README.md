# CNN on CIFAR-10

A simple 3-block convolutional neural network (Conv-BN-ReLU-MaxPool) classifier trained on CIFAR-10 via HuggingFace datasets using PyTorch.

## Usage

```bash
# Local training
python train.py
```

## Model

- **Architecture**: 3x ConvBlock(3→32, 32→64, 64→128) + AdaptiveAvgPool2d + Linear(128→10)
- **Each block**: Conv2d(3x3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)
- **Optimizer**: AdamW (lr=1e-3) with ReduceLROnPlateau (factor=0.5, patience=2)
- **Training**: 10 epochs, gradient clipping at 1.0, early stopping after 5 epochs
- **Data**: 80/20 train/val split of CIFAR-10 (40k train / 10k val / 10k test)
- **Augmentation**: RandomHorizontalFlip(p=0.5)

## Key results

| Metric | Value |
|--------|-------|
| Test accuracy | 71.32% |
| Test loss | 0.8262 |
| Training time | 320s (5.3 min) |
| Model parameters | ~145K |

Top per-class accuracies: truck (85.0%), ship (83.7%), automobile (81.2%). Weakest: horse (57.5%), deer (61.2%), dog (61.6%).

## Gotchas

- Dataset loaded from HuggingFace `uoft-cs/cifar10` — requires internet connection.
- No separate launch.py; run train.py directly.
- Output artifacts: `model.pt`, `metrics.json`, `loss_accuracy_cm.png`, `sample_predictions.png`.
