# Post-LN vs Pre-LN Transformer Comparison

Compare two LayerNorm placement strategies on IWSLT2017 DE-EN translation.

## Architecture Difference

```
Post-LN (original):  x = LayerNorm(x + Dropout(Sublayer(x)))
Pre-LN  (modern):    x = x + Dropout(Sublayer(LayerNorm(x)))
```

Post-LN applies LayerNorm AFTER the residual — gradients must flow through LN, amplifying variance. Pre-LN applies LayerNorm BEFORE sublayers — the residual provides a direct gradient highway, naturally damping variance.

## Quick Start

```bash
# Local smoke test (CPU)
python train.py --ln_type post --max_steps 5 --max_train_pairs 100
python train.py --ln_type pre  --max_steps 5 --max_train_pairs 100

# Colab (two accounts in parallel)
bash launch.sh                    # both accounts
bash fetch.sh                     # download + report
python charts.py                  # comparison visualization
```

## Results (2026-06-14)

Two parallel Colab T4 runs, identical hyperparameters (d_model=256, nhead=4, n_layers=6, warmup=2000, xavier init, Noam LR).

| | Post-LN | Pre-LN |
|---|---|---|
| Steps | 1000 | 1000 |
| Final train loss | 4.860 | **4.812** |
| Final val loss | 4.753 | **4.725** |
| Gradient norm range | 0.9–1.1 | 0.7–0.9 |
| NaN | None | None |
| Training time | 112s | 118s |
| Params | 12.06M | 12.06M |

**Pre-LN wins**: 0.03 lower val_loss and 30% lower gradient norms. Modest but consistent margin — expected since pre-LN's gradient highway enables smoother optimization. In deeper networks (12+ layers) the gap widens significantly.

### First Run (before fix — demonstrates Post-LN instability)

| | Post-LN | Pre-LN |
|---|---|---|
| Steps | 500 | 500 |
| Final train loss | NaN | 5.940 |
| Final val loss | NaN | 5.706 |
| Gradient norms | 14–122 | 1.2–7.8 |
| NaN onset | step 200 | — |

Post-LN went NaN at the Noam warmup peak (lr=0.00442, 6.3× higher than the original paper). Pre-LN trained normally. This is the classic post-LN failure mode.

## Key Learnings

### Noam LR Scaling Trap

The Noam peak LR is `d_model^(-0.5) * warmup^(-0.5)`. Our first run (d=256, warmup=200) peaked at 0.00442. The paper (d=512, warmup=4000) peaks at 0.00070. **6.3× difference.** When scaling down model size, scale warmup proportionally: `warmup ≈ k / d_model` where k ≈ 2,048,000.

### Weight Init Matters

PyTorch's default kaiming_uniform caused step-1 gradient norms of 89. Switching to xavier_uniform + normal embedding init dropped them to 5.1 (**17× reduction**). Standard transformer init is essential.

### Post-LN Gradient Amplification

Post-LN consistently produces 3–15× higher gradient norms than Pre-LN under identical hyperparameters. This is architectural, not configurable away. Pre-LN's `x + Sublayer(LayerNorm(x))` gives a gradient highway through the residual; Post-LN's `LayerNorm(x + Sublayer(x))` doesn't.

### NaN Propagation Pattern

val_loss goes NaN first (no gradient clipping during eval), then train_loss follows 1–2 logging intervals later. By the time train_loss reads NaN, Adam's second moment is already corrupted across all weights. You cannot recover — must restart from last clean checkpoint.

## Model Config

| Param | Value |
|---|---|
| d_model | 256 |
| n_heads | 4 |
| n_layers | 6 |
| d_ff | 512 |
| max_len | 128 |
| vocab | 8000 (word-level) |
| Params | ~12M |

## Data

IWSLT2017 DE-EN from HuggingFace CDN: 206K sentence pairs. Use 25K subset (`--max_train_pairs 25000`) to fit Colab's ~10 min GPU window. Word-level tokenizer — no external dep needed.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Single-file CleanRL-style training. `--ln_type post\|pre` flag. Self-contained: model, data, tokenizer, logging, plotting. |
| `launch.sh` | Deploy to two Colab accounts: `colab` (post-LN) + `cb` (pre-LN) |
| `fetch.sh` | Cron monitor. Tar VM output → download via REST → extract → report. `--summary` for side-by-side comparison. |
| `charts.py` | Post-hoc 4-panel comparison (loss overlay, val loss, LR, bar chart) |
| `gotchas.md` | Field-tested surprises from this project |
