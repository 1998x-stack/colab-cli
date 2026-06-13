# GCN Citation Networks

2-layer Graph Convolutional Network (GCN) for node classification on three standard citation datasets: Cora, CiteSeer, and PubMed.

## Usage

```bash
# Local training (CPU or GPU)
python train.py

# Colab deployment
cb launch.py
```

The script trains sequentially on all three datasets. Per-dataset outputs include CSV metrics, PNG training curves, and a model checkpoint. A comparison dashboard is generated at the end.

## Architecture

- 2-layer GCN (via `torch_geometric.nn.GCNConv`)
- Hidden dimension: 64
- Dropout: 0.5
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Epochs: 100

## Datasets

| Dataset | Nodes | Edges | Classes |
|---------|-------|-------|---------|
| Cora | ~2,708 | ~5,278 | 7 |
| CiteSeer | ~3,327 | ~4,732 | 6 |
| PubMed | ~19,717 | ~44,324 | 3 |

## Gotchas

- Dataset loading tries HuggingFace datasets first (`gcaillaut/{dataset}`), falling back to PyG Planetoid.
- For HF-authenticated datasets, place a HuggingFace token in `/content/.hf_token`.
- Auto-detects Colab, Kaggle, and local environments for output directory.
- GPU-accelerated via PyTorch but small enough to run on CPU.
- Outputs include a `comparison_dashboard.png` with accuracy bar charts and per-dataset info.
