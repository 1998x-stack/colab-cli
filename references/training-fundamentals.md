# Training Fundamentals

Cross-project ML training gotchas verified on Colab T4.

## LR Schedules

### Noam: small d_model + short warmup = dangerously high LR

The Noam formula peaks at `d_model^(-0.5) * warmup^(-0.5)`. With paper params (d=512, warmup=4000): peak = 0.00070. With common "small model" params (d=256, warmup=200): peak = 0.00442 — **6.3x higher**.

Scale warmup proportionally: `warmup = k / d_model` where k ≈ 2,048,000. Quick sanity: `peak = d^(-0.5) * warmup^(-0.5)` — if >0.002, raise warmup.

### Post-LN requires warmup; Pre-LN is robust without it

- Post-LN without warmup → NaN within 100-200 steps
- Pre-LN without warmup → trains fine
- The instability IS the experimental result in comparison studies

## Weight Initialization

### Default PyTorch init can be 17x worse than transformer standard

| Init | Step-1 grad norm |
|------|-----------------|
| kaiming_uniform (default) | 89.0 |
| xavier_uniform + normal embed | 5.1 (17x lower) |

Transformer standard: Linear → `xavier_uniform` with gain `1/sqrt(2)`. Embeddings → `normal(0, d_model^(-0.5))`.

### N(0, 1.0) init causes catastrophic divergence

Initial loss 2051 (expected: 2.30 for K=10). Always run the "loss at init" sanity check:
```python
loss = F.cross_entropy(model(randn_input), randn_labels)
assert abs(loss.item() - 2.3026) < 0.5  # K=10 → -ln(1/10) = 2.3026
```

### Overfit single batch before full training

Verify model can memorize 16 fixed samples to near-zero loss (<0.01 in 200 steps). Remove dropout, weight decay, and augmentation. Catches ~80% of structural bugs in 2 minutes.

## NaN Diagnosis

### Propagation order: val_loss first, train_loss 1-2 intervals later

NaN shows in validation first (no gradient clipping protection during eval). If val_loss=NaN, stop immediately — weights are already corrupted.

### NaN is contagious through Adam's second moment

Once one weight becomes NaN, Adam's `v_t` accumulates NaN. Within 1-2 steps, all weights infected. Must restart from last clean checkpoint.

### Gradient clipping at 1.0 masks but doesn't prevent NaN

Clipping cuts 98%+ of gradient magnitude but doesn't fix the underlying instability. Fix the LR schedule and initialization — don't rely on aggressive clipping as a band-aid.

## CUDA / AMP

### AMP on T4: 2-3x speedup

```python
# PyTorch 2.11 API (torch.cuda.amp is deprecated)
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    logits = model(src, tgt_in, src_mask, tgt_mask)
    loss = criterion(...)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

### T4: FP16 yes, BF16 no

T4 SM 7.5 < 8.0 → no bfloat16 hardware support. Use float16 for tensor cores, float32 for precision-sensitive ops.

### CUDA timing: perf_counter without synchronize() is 3-15x wrong

Always use `torch.cuda.Event` for GPU benchmarks:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... GPU work ...
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

## Cross-Entropy + Permute: 17-118x slower for LLM logits

`F.cross_entropy(logits.permute(0, 2, 1), targets)` triggers slow reduction kernels. Fix:
```python
# ❌ 51.6 ms
loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
# ✅ 0.45 ms (114x faster)
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
```

## view() vs reshape()

After `.permute()`, `.transpose()`, or `.T`: `.view()` crashes (`RuntimeError`), `.reshape()` copies internally when needed — always safe.

## CUDA 12.8 / PyTorch 2.11 Fixed Traps

The following "classic" issues were NOT observed on this stack:

| Trap | Old expectation | Actual (CUDA 12.8) |
|------|----------------|--------------------|
| Implicit `.contiguous()` copies | 2-10× slowdown | 1.0-1.1× |
| `index_select` 2-6× slower for 2D+ | 2-6× | 1.0-1.2× |
| FP16 eps=1e-8 NaN within 200 steps | NaN | No NaN in 500 steps |
| CUDA first-call tax ~1.6s | 1.6s | ~389ms |

Always verify "known" traps on your target CUDA/PyTorch version.
