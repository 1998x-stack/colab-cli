# Sequence to Sequence Learning with Neural Networks

**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google, 2014)
**arXiv:** [1409.3215](https://arxiv.org/abs/1409.3215)

## TL;DR

First end-to-end neural approach to sequence-to-sequence learning. Uses two deep LSTMs — one encoder that reads the input sequence into a fixed-dimensional vector, and one decoder that generates the output sequence from that vector. The key insight: **reversing the source sentence order** dramatically improves performance by introducing short-term dependencies between source and target.

## Architecture

```
Source:  A   B   C   <EOS>           → encoder LSTM
         ↓   ↓   ↓     ↓
        [LSTM→LSTM→LSTM→LSTM]  (4 layers, 1000 cells)
                          ↓
                    fixed vector v
                          ↓
        [LSTM→LSTM→LSTM→LSTM]  (4 layers, 1000 cells)
         ↓   ↓   ↓   ↓   ↓
        <SOS> W   X   Y   Z      ← decoder LSTM
```

- **Encoder:** 4-layer deep LSTM, reads input sequence (reversed), outputs final hidden state
- **Decoder:** 4-layer deep LSTM, conditioned on encoder's final hidden state, generates output autoregressively
- **Embeddings:** 1000-dimensional, learned jointly
- **Vocabulary:** 160K source tokens, 80K target tokens (frequent words + 40K BPE subwords on source)
- **Total parameters:** 384M (encoder + decoder, including embeddings)

### Why Deep LSTMs?

Paper ablations show: 4-layer LSTM > 1-layer LSTM (+2 BLEU on test). Depth matters for generalization.

### Why Reverse Source?

```
Normal:    A B C → α β γ    (long path from A to α)
Reversed:  C B A → α β γ    (A is next to α, C near γ — short-term dependencies)
```

Reversing the source introduces many short-term dependencies between source and target, making SGD optimization easier. Gains ~9 BLEU points on long sentences.

## Training

| Parameter | Value |
|-----------|-------|
| Dataset | WMT'14 English→French (12M sentence pairs, 348M French words) |
| Optimizer | SGD with momentum |
| Learning rate | 0.7, halved every epoch after epoch 5 |
| Batch size | 128 sentences |
| Gradient clipping | 5.0 (sum-of-gradients norm) |
| Epochs | 7.5 total |
| Initialization | Uniform(-0.08, 0.08) |
| Regularization | None (no dropout in base; dropout used in ensemble) |
| Loss | Cross-entropy, masked on padding |

## Inference (Beam Search)

- **Beam size:** 12 (best tradeoff of speed vs BLEU)
- **Penalty:** No length normalization in base model
- **Ensemble:** 5 LSTMs + beam 12 → 37.5 BLEU (SOTA at the time)

## Results

| Method | BLEU |
|--------|------|
| LSTM (no reverse, beam 1) | 25.9 |
| LSTM (reverse, beam 1) | 28.0 |
| **LSTM (reverse, beam 12)** | **34.81** |
| LSTM + SMT rescoring (1000-best) | 36.5 |
| 5× LSTM ensemble + beam 12 | 37.5 |
| Phrase-based SMT (baseline) | 33.3 |

## Key Properties

- **Long sentences:** LSTM degrades gracefully on >60 word sentences; reversing source eliminates the penalty entirely
- **Word order sensitivity:** Active/passive transformations preserved (embeddings cluster by meaning, not surface form)
- **Fixed-dimension bottleneck works:** All information compressed through a single ~1000-dim vector — surprisingly effective

## Our Implementation (Colab T4, scaled-down)

Paper → Our version tradeoffs:
- WMT'14 EN→FR (12M pairs) → Multi30k EN→DE (29K pairs)
- 384M params → ~5M params (2 layers, 256 hidden)
- 7.5 epochs × 12M pairs → 20 epochs × 29K pairs
- 160K/80K vocab → 8K BPE vocab
- Beam 12 → Beam 5 (speed)
- Reverse source: ✅ (flag toggle)
- Beam search: ✅ (sacrebleu-powered)
