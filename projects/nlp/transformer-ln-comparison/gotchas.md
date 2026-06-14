# Post-LN vs Pre-LN Transformer — Gotchas

Field-tested surprises from the transformer-ln-comparison project (2026-06-14).

## Training stability

### Noam scheduler peak LR was 6.3x higher than the original paper

**Symptom:** Post-LN training goes NaN at step 200 (exactly at the warmup peak). Pre-LN trains fine.

**Root cause:** The Noam peak LR is `d_model^(-0.5) * warmup^(-0.5)`. Our default (d_model=256, warmup=200) gives peak LR = 0.00442. The original paper (d_model=512, warmup=4000) gives peak LR = 0.00070. That's 6.3x higher.

**Why only post-LN breaks:** Post-LN's `LayerNorm(x + Sublayer(x))` amplifies gradient variance because the gradient must flow through LayerNorm after each residual. Pre-LN's `x + Sublayer(LayerNorm(x))` provides a direct gradient highway through the residual connection, naturally damping variance.

**Fix:** Three changes that together dropped step-1 gradient norm from 89 → 5 (17x reduction):
1. Warmup 200 → 2000 (lowers peak LR proportionally)
2. Add `--lr_scale` parameter to NoamScheduler
3. Add proper weight init: `xavier_uniform` for Linear, `normal(std=d_model^(-0.5))` for Embedding

**Formula check before training:**
```python
d_model = 256
warmup = 200
peak_lr = (d_model ** -0.5) * (warmup ** -0.5)  # 0.00442 — too high
# Compare to paper: d=512, warmup=4000 → peak=0.00070
```

### Weight initialization matters more than you'd think

PyTorch's default Linear init (kaiming_uniform, fan-in mode) caused 17x higher initial gradients than xavier_uniform for this transformer. The embedding init matters too — `normal(std=d_model^(-0.5))` is the standard transformer init.

**Before (default init):** Step 1 loss=62.6, grad=89.0
**After (xavier + normal embed):** Step 1 loss=9.6, grad=6.0

### Gradient norms are 3-15x higher in post-LN

Even with identical hyperparameters, post-LN consistently shows 3-15x higher gradient norms than pre-LN. This is architectural — not a bug. Monitor gradient norms during training; if they exceed ~50 in post-LN, the run will likely NaN.

**Observed ranges (stable run, warmup=2000):**
- Post-LN: grad 3-8
- Pre-LN: grad 1-3

### NaN propagates val_loss first, then train_loss one interval later

At the inflection point, `val_loss=nan` appears one logging interval before `train_loss=nan`. The model weights are already corrupted by the time train_loss goes NaN. If you see val_loss=NaN, stop training — continuing just wastes GPU time.

## Data & tokenization

### Word-level tokenizer: OOV rate is low on IWSLT but high on open-domain text

With vocab=8000 on 25K IWSLT pairs (~40K unique words total), OOV rate stays under 5%. For open-domain text, expect 15-20% OOV. This is fine for a comparison experiment but would hurt BLEU on a production system.

### First session caches tokenized data — second session is 45s faster

The first Colab session downloads the 18MB IWSLT ZIP and tokenizes all pairs (~30s on CPU, 2s on T4 GPU). The tokenized cache (`/content/iwslt_data/cache/*.pt`) vanishes when the session ends. If you need multiple runs, either:
- Pre-tokenize and re-upload the cached `.pt` files
- Accept the 30s tokenization tax on each new session
- Use a warmup session to cache, then re-provision

## fetch.sh

### bash 3.2 (macOS default) doesn't support `${var^^}` uppercase expansion

```bash
# WRONG (bash 4.0+):
echo "${label^^}"    # postln → POSTLN

# RIGHT (bash 3.2):
echo "$label" | tr '[:lower:]' '[:upper:]'
```

### Python inline NaN comparison needs quoting

When embedding CSV values in Python `-c` expressions from bash, "nan" strings must be quoted:

```bash
# WRONG — bash expands nan as a bare word
python3 -c "print('better' if $val_a < $val_b else 'worse')"
# NameError: name 'nan' is not defined

# RIGHT — quote the values
python3 -c "a='$val_a'; b='$val_b'; ..."
```
