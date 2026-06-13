# AlexNet on Imagenette

AlexNet (Krizhevsky et al., NeurIPS 2012) trained on Imagenette-160 (10-class subset of ImageNet, 160px) using PyTorch, with a 4-experiment ablation study comparing baseline, no-dropout, no-data-augmentation, and reduced-width configurations.

## Usage

```bash
# Local training (select experiment IDs 1-4)
python train.py --exp_ids 1,2,3,4

# Colab deployment
# Create /content/exp_ids.txt with experiment IDs, then:
cb launch.py
```

## Model

- **Architecture**: AlexNet with 5 conv layers + 3 FC layers, AdaptiveAvgPool2d(6) for 128x128 input
- **Initialization**: He (Kaiming) init (original N(0,0.01) doesn't converge at 128x128)
- **Input**: 128x128 crops from 160px images
- **Optimizer**: SGD (lr=0.001, momentum=0.9, weight_decay=0.0005) with ReduceLROnPlateau
- **Epochs**: 20 (max), early stopping when LR falls below 1e-6
- **Evaluation**: 10-view test (5 crops + flips)
- **Experiments**: (1) Baseline, (2) No Dropout, (3) No Data Augmentation, (4) Reduced Width (0.5x)

## Gotchas

- LRN layers from the original paper are omitted (obsolete with BatchNorm).
- He initialization is critical — the paper's N(0,0.01) init fails on 128x128 input.
- PCA color augmentation (Fancy PCA) is fitted on a 500-sample subset before training.
- launch.py requires `/content/exp_ids.txt` and optionally `/content/hf_token`.
