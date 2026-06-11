# Transformer (Attention Is All You Need) on IWSLT'14 De→En — Design Spec

**Date:** 2026-06-11
**Status:** designing

## Goals

1. Implement the full Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) in PyTorch
2. Train on IWSLT'14 German→English (160K sentence pairs) — a real MT task that fits Colab T4
3. Run 3 ablation experiments in parallel across 3 Colab GPU accounts with checkpoint-resume for session survival
4. Produce paper-equivalent charts: loss curves, BLEU curves, ablation comparison, attention maps, positional encoding comparison
5. Stream checkpoints every epoch to survive VM death (~10 min T4 sessions)

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | IWSLT'14 De→En (160K pairs) | Real MT task, BLEU measurable, fits T4 VRAM, short sentences |
| Tokenizer | BPE (32K vocab), shared source/target | Paper used BPE; shared vocab enables weight tying |
| Model scale | Base (d_model=512, 6L+6L, 8 heads, ~65M params) | Paper's base model; fits T4 15.6 GB with batch_size=64 |
| Positional encoding | Learned (baseline), sinusoidal (ablation) | Paper says they're equivalent; learned is simpler code |
| Weight tying | Shared encoder/decoder input + output projection | Paper §3.4 mentions this; reduces params by ~15M |
| Label smoothing | Omitted | Simplification; paper used 0.1 but not critical for ablation comparison |
| Optimizer | Adam, β₁=0.9, β₂=0.98, ε=10⁻⁹ | Paper §5.3 exact values |
| LR schedule | Warmup 4000 steps → 1/sqrt(step) decay | Paper §5.3 formula |
| Beam search | Beam size 4 for BLEU eval | Standard for IWSLT; fast enough on T4 |
| Validation | 80/20 random split of IWSLT train set | IWSLT test sets are small; split gives meaningful val BLEU |
| Eval metric | SacreBLEU (detokenized) | Reproducible, standard |
| Session strategy | 1-experiment-per-account, checkpoint-resume chain | Parallel throughput; independent chains = no coupling |
| Accounts | `colab` (baseline), `cb` (fixed_pe), `clb` (heads_1) | Three free GPU slots; parallel execution |
| Session names | `transformer-baseline`, `transformer-fixedpe`, `transformer-heads1` | One per account per experiment |
| Project dir | `projects/transformer_iwslt/` | Follows repo conventions (snake_case) |

## Architecture

```
Transformer (~65M params, d_model=512, 6L encoder + 6L decoder, 8 heads)

Input: German tokens (BPE, 32K vocab) → Embedding + Learned PE
Output: English tokens (autoregressive, shared vocab + embedding)

Encoder:
  Input Embedding (vocab=32000, d_model=512)
  + Learned Positional Encoding (max_len=512, d_model=512)
  ├── EncoderLayer ×6
  │   ├── Multi-Head Self-Attention (h=8, d_k=64, d_v=64)
  │   ├── Dropout 0.1 → Add & LayerNorm
  │   ├── Position-wise FFN (512→2048→512, ReLU)
  │   └── Dropout 0.1 → Add & LayerNorm
  └── Output: (batch, src_len, 512)

Decoder:
  Output Embedding (shared with encoder)
  + Learned Positional Encoding
  ├── DecoderLayer ×6
  │   ├── Masked Multi-Head Self-Attention (h=8, d_k=64, d_v=64)
  │   ├── Dropout 0.1 → Add & LayerNorm
  │   ├── Cross-Attention (Q=decoder, K/V=encoder output)
  │   ├── Dropout 0.1 → Add & LayerNorm
  │   ├── Position-wise FFN (512→2048→512, ReLU)
  │   └── Dropout 0.1 → Add & LayerNorm
  └── Linear(512, 32000) + Softmax

Scale dot-product attention by 1/sqrt(d_k) before softmax.
Padding mask: <pad> token ignored in attention via -inf mask.
Causal mask: upper triangular -inf for decoder self-attention.
```

## 3 Experiments (1 per account, parallel)

| # | Exp ID | Account | Session name | Diff from baseline | Expected Δ |
|---|---|---|---|---|---|
| 1 | `baseline` | `colab` | `transformer-baseline` | Full Transformer, learned PE, 8 heads, d_k=64 | Target: >25 sacreBLEU |
| 2 | `fixed_pe` | `cb` | `transformer-fixedpe` | Sinusoidal positional encoding (paper Eq. 3-5) instead of learned parameters | Small drop, especially on longer sentences |
| 3 | `heads_1` | `clb` | `transformer-heads1` | 1 attention head instead of 8 (d_k=512, d_v=512, keeping same total QKV dimension) | Noticeable drop — matches paper §5.4 |

Each experiment: 20 epochs, batch_size=64, max_tokens=4096 per batch.

## Training Hyperparameters

| Parameter | Value (matches paper) |
|---|---|
| Optimizer | Adam, β₁=0.9, β₂=0.98, ε=10⁻⁹ |
| Initial LR | 0.0001 |
| LR schedule | Warmup over 4000 steps, then 1/sqrt(step) decay |
| Batch size | 64 (sentences) |
| Max tokens/batch | 4096 (dynamic batching) |
| Epochs | 20 |
| Loss | Cross-entropy, ignore <pad> |
| Dropout | 0.1 (attention weights, FFN, embeddings) |
| Label smoothing | Omitted |
| Gradient clipping | 1.0 (max norm) |
| Beam size | 4 |
| Max decode length | 128 tokens |

## Code Structure

```
projects/transformer_iwslt/
├── model.py              # Model definition
│   ├── MultiHeadAttention, PositionwiseFFN
│   ├── EncoderLayer, DecoderLayer
│   ├── Encoder, Decoder, Transformer
│   ├── SinusoidalPE (for fixed_pe experiment)
│   └── build_transformer(config) factory
│
├── train.py              # Training loop + eval
│   ├── IWSLT'14 De→En dataset loading (torchtext or direct download)
│   ├── BPE tokenizer (huggingface tokenizers, 32K vocab, shared)
│   ├── Training loop with checkpoint save/resume
│   ├── BLEU evaluation (beam search, sacrebleu)
│   ├── --exp_id flag to select experiment config
│   └── Metrics streaming to metrics.jsonl
│
├── launch.py             # Colab bootstrap
│   ├── pip install deps
│   ├── Upload checkpoint resume if exists
│   └── Detached spawn of train.py
│
├── check_progress.py     # Local monitoring
│   ├── Session alive check (colab sessions + colab status)
│   ├── Process alive check (pgrep python)
│   ├── Log tail (last 20 lines)
│   ├── Epoch count from metrics.jsonl
│   └── Alert if epoch count > 18 → flag near-completion
│
├── charts.py             # Post-hoc result generation (runs locally)
│   ├── Loss curves (3 experiments overlaid)
│   ├── BLEU over epochs
│   ├── Ablation comparison bar chart
│   ├── Attention heatmap visualization
│   └── Positional encoding comparison (learned vs sinusoidal)
│
└── checkpoint.py         # Checkpoint save/load helpers
    ├── save_checkpoint(model, optimizer, scheduler, epoch, path)
    ├── load_checkpoint(path) → model, optimizer, scheduler, epoch
    └── upload_checkpoint_to_session(account, session, local_path)
```

## Session Orchestration (1-experiment-per-account, parallel)

Each experiment runs its own independent checkpoint-resume chain:

```
Per-experiment loop (until epoch 20 reached):
  1. Provision: <account> new --gpu T4 -s transformer-<exp_id>
  2. Upload: model.py, train.py, launch.py, checkpoint.py
     + checkpoint_epoch{N}.pt if resuming
  3. Launch: <account> exec -f launch.py --timeout 120
     (launch.py spawns train.py --exp_id <exp_id> --resume /content/checkpoint_epoch{N}.pt)
  4. Monitor: CronCreate every 5 min, check_progress.py against session
  5. On session death (detected by cron):
     - Download latest checkpoint + metrics.jsonl immediately
     - Go to step 1 with next session on same account
  6. On epoch 20 reached:
     - Download final model, all checkpoints, metrics.jsonl
     - Stop session, mark experiment done
```

Since all 3 run in parallel:

| Account | Session name | Cron job (every 5 min) |
|---|---|---|
| `colab` | `transformer-baseline` | `colab exec -s transformer-baseline -f check_progress.py --timeout 15` |
| `cb` | `transformer-fixedpe` | `cb exec -s transformer-fixedpe -f check_progress.py --timeout 15` |
| `clb` | `transformer-heads1` | `clb exec -s transformer-heads1 -f check_progress.py --timeout 15` |

Estimated ~6-8 sessions per experiment × 3 experiments = ~18-24 sessions total. Wall time ~2-3 hours (parallel execution).

## Session Lifetime Budget

| Metric | Value |
|---|---|
| IWSLT'14 train set | ~160K sentence pairs |
| Tokens/epoch | ~3.2M (avg ~20 tokens/sentence) |
| Per-epoch time (T4, 65M params, bs=64) | ~4 min |
| 20-epoch experiment | ~80 min training |
| Per-session survival | ~10 min (2-3 epochs) |
| Sessions needed per experiment | ~7-10 sessions |
| Checkpoints saved per experiment | ~20 (one per epoch) |

## Output Artifacts

Per-experiment raw data:
- `metrics.jsonl` — per-epoch: train_loss, val_loss, bleu, lr, tokens_processed, wall_time_s
- `config.json` — full hyperparameters for reproducibility
- `checkpoints/` — last 3 epochs (.pt files)

Post-hoc charts (generated locally by `charts.py`):

| Chart | Content |
|---|---|
| `loss_curves.png` | Train/val loss vs epochs, 3 experiments overlaid |
| `bleu_curves.png` | SacreBLEU vs epochs, 3 experiments overlaid |
| `ablation_bars.png` | Final BLEU bar chart, 3 experiments side by side |
| `attention_heads.png` | 8-head attention weights from baseline encoder layer 1, sample German sentence |
| `position_encoding.png` | Learned PE cosine similarity matrix (baseline) vs sinusoidal PE heatmap (fixed_pe) side by side |
| `results_summary.md` | Table: final BLEU, best epoch, training time per experiment |

## Job Flow (Detailed)

```bash
# ═══ Parallel setup: all 3 accounts at once ═══

# 1. Provision all sessions
colab new --gpu T4 -s transformer-baseline
cb new --gpu T4 -s transformer-fixedpe
clb new --gpu T4 -s transformer-heads1

# 2. Upload to each session (in parallel)
colab upload projects/transformer_iwslt/model.py /content/model.py
colab upload projects/transformer_iwslt/train.py /content/train.py
colab upload projects/transformer_iwslt/launch.py /content/launch.py
colab upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py

cb upload projects/transformer_iwslt/model.py /content/model.py
cb upload projects/transformer_iwslt/train.py /content/train.py
cb upload projects/transformer_iwslt/launch.py /content/launch.py
cb upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py

clb upload projects/transformer_iwslt/model.py /content/model.py
clb upload projects/transformer_iwslt/train.py /content/train.py
clb upload projects/transformer_iwslt/launch.py /content/launch.py
clb upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py

# 3. Launch all (in parallel)
colab exec -s transformer-baseline -f launch.py --timeout 120
cb exec -s transformer-fixedpe -f launch.py --timeout 120
clb exec -s transformer-heads1 -f launch.py --timeout 120

# ═══ Monitoring: 3 cron jobs ═══
# Cron job A: colab exec -s transformer-baseline -f check_progress.py --timeout 15  (every 5 min)
# Cron job B: cb exec -s transformer-fixedpe -f check_progress.py --timeout 15     (every 5 min)
# Cron job C: clb exec -s transformer-heads1 -f check_progress.py --timeout 15    (every 5 min)

# ═══ On session death (detected by cron) ═══
# Download checkpoint immediately:
colab download /content/checkpoints/checkpoint_epoch{N}.pt ./projects/transformer_iwslt/output-baseline/checkpoints/
colab download /content/metrics.jsonl ./projects/transformer_iwslt/output-baseline/

# Re-provision and resume:
colab new --gpu T4 -s transformer-baseline
colab upload projects/transformer_iwslt/output-baseline/checkpoints/checkpoint_epoch{N}.pt /content/checkpoint_epoch{N}.pt
# ...upload code files again...
colab exec -s transformer-baseline -f launch.py --timeout 120
# (launch.py passes --resume /content/checkpoint_epoch{N}.pt to train.py)

# ═══ Final download (epoch 20 reached) ═══
colab download /content/metrics.jsonl ./projects/transformer_iwslt/output-baseline/
colab download /content/config.json ./projects/transformer_iwslt/output-baseline/
tar -czf /content/checkpoints.tar.gz -C /content checkpoints/
colab download /content/checkpoints.tar.gz ./projects/transformer_iwslt/output-baseline/
colab stop -s transformer-baseline

# ═══ Post-hoc: run charts locally ═══
python projects/transformer_iwslt/charts.py
```

## Success Criteria

- [ ] Baseline Transformer reaches >25 sacreBLEU on IWSLT'14 De→En test (paper base model: 27.3)
- [ ] Fixed PE ablation shows measurable BLEU difference from learned PE
- [ ] 1-head ablation shows clear performance degradation vs 8-head baseline
- [ ] All 6 PNG artifacts render correctly
- [ ] metrics.jsonl has complete per-epoch data for all 3 experiments
- [ ] Checkpoint-resume works: model resumes from exact epoch with no loss discontinuity
- [ ] Training curves show expected warmup behavior (loss drops after ~4000 steps)
- [ ] Multi-head attention visualization shows diverse attention patterns across heads
