# FastText PyTorch

FastText classifier implemented from scratch in PyTorch — subword character n-gram embeddings + averaged document vectors + linear classifier.

Based on: *"Enriching Word Vectors with Subword Information"* (Bojanowski et al., 2017)

## Architecture

```
document → tokenize → word_emb + mean(ngram_embs) → mean across words → linear → class
```

Each word is represented as its word embedding + the average of its character n-gram embeddings (e.g., "hello" → `<he`, `hel`, `ell`, `llo`, `lo>` etc.). This captures subword morphology — prefixes, suffixes, character patterns.

## Dataset

**[AG News](https://huggingface.co/datasets/fancyzhx/ag_news)** — 120k news articles, 4 classes (World, Sports, Business, Sci/Tech).

## Quick start

```bash
# Local test (tiny: 30k docs, 50d, 3 epochs, ~5 min on M-series Mac)
python train.py --size tiny --out-dir ./output

# Full training (120k docs, 100d, 5 epochs)
python train.py --size small --out-dir ./output

# Custom
python train.py --embed-dim 200 --epochs 10 --batch-size 128
```

## Colab CPU

```bash
# 1. Provision
colab new -s fasttext-train

# 2. Upload files
colab upload train.py /content/train.py
colab upload launch.py /content/launch.py

# 3. Upload HF token (optional — for private datasets)
colab upload ~/.huggingface/token /content/.huggingface_token

# 4. Launch (installs deps + spawns training detached)
colab exec -f launch.py --timeout 120

# 5. Monitor via cron (fetch.sh pulls outputs every 4 min)
# See fetch.sh — set up via CronCreate
```

## Size presets

| Preset | embed_dim | vocab | ngram_buckets | epochs | docs | Params | Est. CPU time |
|--------|-----------|-------|---------------|--------|------|--------|---------------|
| `cpu` (default) | 50 | 30k | 200k | 5 | 120k | 11.3M | ~15-20 min |
| `tiny` | 50 | 20k | 500k | 3 | 30k | 25.1M | ~10 min |
| `small` | 100 | 50k | 2M | 5 | 120k | 202.7M | ~45-60 min (GPU recommended) |

## Output

```
/content/fasttext-pytorch-output/
├── logs/train.log          # Timestamped training log
├── metrics.csv             # Per-epoch metrics (loss, accuracy, time)
├── pngs/training_curves.png  # Multi-panel visualization
├── checkpoints/model_epoch01.pt  # Epoch checkpoints
└── summary.json            # Final metrics + config
```

## Metrics

| Log line | `[HH:MM:SS] Ep 1/5 \| Batch 500 \| loss=0.8523 \| avg100=0.8912 \| lr=0.095000 \| elapsed=180s` |
|----------|------|
| **CSV** | `epoch,train_loss,train_acc,test_loss,test_acc,elapsed_s,lr` |
| **PNG** | Loss curve, test accuracy over time, LR schedule, loss distribution |

## Key design decisions

- **CleanRL style**: single file, argparse config, no trainer classes, direct loops
- **FNV-1a hash** for deterministic n-gram bucket assignment across runs
- **Mean pooling** for document vectors (matching original fastText supervised mode)
- **Gradient clipping** (5.0) for training stability
- **No pre-trained embeddings** — trained from scratch on AG News
