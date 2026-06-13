# RNN-IMDB

Bidirectional LSTM sentiment classifier on the IMDB dataset (25K training, 25K test reviews) using PyTorch with HuggingFace datasets.

## Usage

```bash
# Local training
python train.py
```

This project does not include a `launch.py` for Colab deployment.

## Key results

| Metric | Value |
|--------|-------|
| Test accuracy | 0.7764 |
| Test F1 | 0.7824 |
| Test loss | 0.4952 |
| Best epoch | 3 (val_acc=0.7826, val_f1=0.8780) |
| Total train time | 94.2s on CUDA |
| Early stopping | Epoch 6 (patience=3) |

The model overfits after epoch 3 -- train loss continues decreasing (epoch 6: 0.1052) while val loss climbs (epoch 6: 1.3565). Best checkpoint selected by lowest val loss at epoch 3.

## Gotchas

- Vocabulary built from training split only, using the top 25K words by frequency with `<pad>` (index 0) and `<unk>` (index 1).
- Reviews are truncated/padded to 500 tokens. Longer reviews lose tail content.
- The model uses a bag-of-words-level tokenizer (whitespace split, lowercased) rather than subword tokenization, so rare/OOV words map to `<unk>`.
- Binary cross-entropy loss with sigmoid output (not cross-entropy with softmax).
- Learning rate is halved by `ReduceLROnPlateau` when val loss plateaus (factor=0.5, patience=1).
- Best model saved by lowest val loss, not highest val accuracy or F1.
- F1 computed as micro-F1 (global TP/FP/FN), not macro-averaged.
