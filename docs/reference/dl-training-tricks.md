# Deep Learning Training Tricks — Field-Tested Techniques & Non-Obvious Gotchas

Date: 2026-06-14 | Colab T4 | PyTorch 2.11

Synthesis of Karpathy's methodology, Google Tuning Playbook, fast.ai, EleutherAI, Chinese ML community (Zhihu, CSDN), and our own Colab-verified gotchas. Focus: things that are non-obvious, surprising, or silently wrong — not textbook basics.

---

## 1. Karpathy's Methodology: The Debugging Pipeline

### 1.1 Overfit a single batch first (MANDATORY first step)

Before any hyperparameter tuning, verify the model can memorize 2-16 examples to near-zero loss. Remove all regularization (dropout, weight decay, augmentation) for this test.

**Why non-obvious:** Most skip this and spend hours debugging "the model." If it can't memorize 5 examples, something is **structurally broken** — wrong loss function, bad initialization, broken data pipeline, gradient flow issue. This 5-minute test catches them all.

**Colab T4:** Runs in seconds on any model.

### 1.2 Loss at initialization = sanity check

For K-class softmax classification, initial loss must be approximately `-ln(1/K)`:
- CIFAR-10: ~2.30
- ImageNet-1k: ~6.91
- Binary classification: ~0.69

If actual initial loss deviates significantly, initialization or loss function is wrong.

### 1.3 Karpathy's 6 classic NN mistakes

1. Didn't try to overfit a single batch first
2. Forgot to toggle `model.train()` / `model.eval()` (affects BatchNorm, Dropout)
3. Forgot `.zero_grad()` before `.backward()` (gradients accumulate by default)
4. Passed softmaxed outputs to a loss expecting raw logits (`CrossEntropyLoss` already has softmax)
5. Didn't use `bias=False` for Conv/Linear before BatchNorm (redundant parameters)
6. Confused `view()` and `permute()` — `view()` requires contiguous memory; use `reshape()` for safety

### 1.4 LR too high? Loss > 3× initial → abort

Karpathy's heuristic: if cost ever exceeds 3× the initial cost, the learning rate is too high. Break out early — the run won't recover.

### 1.5 "Become one with your data"

Spend hours inspecting raw data before writing model code. Look for duplicates, corrupt samples, label errors, class imbalance. Both Karpathy and HuggingFace (Victor Sanh) independently stress this as the highest-leverage activity.

---

## 2. Learning Rate & Optimization

### 2.1 OneCycleLR: 2-10× faster convergence

**Technique:** LR ramps up linearly, then cosine-anneals to near-zero. Momentum cycles inversely (high→low→high). PyTorch: `torch.optim.lr_scheduler.OneCycleLR`.

**Why it works:** Large LRs are strong regularizers. Cycling momentum balances the large LR, preventing divergence. This achieves super-convergence — SOTA accuracy in 1/10 the iterations.

**Colab-critical:** With Colab's ~10 min GPU window, OneCycleLR can be the difference between completing training and not.

**Magnitude:** 2-10× convergence speedup.

### 2.2 LR Finder: find the best LR in one epoch

**Technique:** Start with tiny LR, exponentially increase every batch, plot loss. Pick the LR at the steepest descent point (roughly 1/10 of the LR where loss explodes).

```python
# fast.ai-style LR finder pattern
for batch in dataloader:
    lr = lr_start * (lr_mult ** step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    loss.backward(); optimizer.step()
    losses.append(loss.item())
    if loss > min_loss * 4: break  # stop when diverging
```

**Why non-obvious:** A single epoch reveals the optimal LR. Beats guessing or grid search by 3-5× trial reduction.

### 2.3 AdamW over Adam — weight decay decoupling matters

Adam's L2 regularization interacts badly with adaptive gradients — it's "technically wrong." AdamW decouples weight decay from the adaptive update, fixing cross-layer effective LR imbalance. This is the single most underappreciated optimizer fix of the past 5 years.

```python
# Wrong: Adam with weight_decay silently does L2 regularization
optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=0.01)

# Correct: weight decay applied separately from gradient
optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
```

**Already documented:** `docs/reference/model-gotchas.md` (Noam LR trap)

### 2.4 LR warmup prevents "unrecoverable loss space"

Without warmup, large initial LRs can push the optimizer into a region from which it never recovers — even after later LR decay. Warmup gives the optimizer time to find a stable basin before committing to large steps.

**Transformer-specific:** Post-LN Transformers diverge within 100-200 steps without warmup. Pre-LN is more robust but still benefits. Warmup + gradient clipping serve the same purpose: controlling `η_t · ||g_t||`.

**Already documented:** `docs/reference/model-gotchas.md` (Post-LN vs Pre-LN stability)

### 2.5 Batch size × LR linear scaling

Doubling batch size → double learning rate (within stability bounds). The empirical optimal LR is approximately proportional to batch size.

### 2.6 EMA of weights: free 0.5-1% accuracy

Maintain a shadow copy of weights with exponential moving average (β=0.999-0.9999). Use shadow weights at eval time. Implementation: ~5 lines. Benefit: zero training cost, consistent 0.5-1% accuracy improvement. The best ROI among all regularization techniques.

### 2.7 SWA (Stochastic Weight Averaging): free 1-2%

Average checkpoints from the last ~25% of training. PyTorch native: `torch.optim.swa_utils.AveragedModel`. Finds flatter minima with better generalization. One of the few techniques that is both trivial to implement and clearly beneficial.

---

## 3. Training Stability

### 3.1 Gradient clipping is a safety net, not a fix

`clip_grad_norm_(max_norm=1.0)` is essential for RNNs and Transformers. But:

- If >50% of updates are being clipped, you're not clipping — you're doing weird LR decay. Lower the LR.
- Clipping at 1.0 with 50-120 gradient norms cuts 98%+ of gradient magnitude. This masks NaN temporarily but doesn't fix the root cause.
- **Fix the LR schedule and initialization** — don't rely on aggressive clipping as a band-aid.

**Already documented:** `docs/reference/model-gotchas.md` (gradient clipping analysis)

### 3.2 NaN is contagious through Adam's second moment

Once a single weight becomes NaN, Adam's `v_t` accumulates NaN. Within 1-2 steps, all weights propagate NaN through the EMA. No recovery possible — must restart from last clean checkpoint.

### 3.3 Val loss NaN appears before train loss NaN

NaN shows in validation first (no gradient clipping protection, larger effective batch). Train loss stays finite for 1-2 intervals because clipping masks the NaN in backward pass. **If val_loss=NaN, stop immediately** — weights are already corrupted.

**Already documented:** `docs/reference/model-gotchas.md` (NaN diagnosis)

### 3.4 Gradient norms are leading indicators; loss is lagging

Log unclipped gradient norms at each step. Spikes in gradient norm precede loss spikes by 5-10 steps. Loss is a trailing indicator of instability.

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
# log total_norm BEFORE the actual clipping call
```

### 3.5 End-of-training gradient explosion (Defazio, Meta FAIR, 2025)

Weight decay + LayerNorm + cosine LR causes gradient norm to spike at the end of training (often 2×). Root cause: LayerNorm forces gradients orthogonal to weights; weight decay drives gradient/weight ratio toward a fixed point that explodes as LR→0. Fix: scale weight decay with LR: `λ̂_t = λ * η_t / η_max`.

### 3.6 Small batch size = oscillating loss

Below a minimum batch size, gradient variance dominates. If loss oscillates, increase effective batch size via gradient accumulation — don't touch the LR first.

---

## 4. Regularization & Augmentation

### 4.1 Label smoothing: prevents overconfidence AND attention collapse

`nn.CrossEntropyLoss(label_smoothing=0.1)`. Not just regularization — in Transformers, it prevents "attention collapse" where attention concentrates on too few tokens. Clean datasets: ε=0.05. Noisy datasets: ε=0.1-0.2.

### 4.2 Data augmentation: turns overfitting into underfitting

Karpathy's framing: data augmentation converts an overfitting problem into an underfitting problem. Then you fix underfitting with a larger model and more training. This is the right mental model — don't treat augmentation as a "regularization knob."

**But:** On short training budgets (<90 epochs), augmentation can hurt by slowing convergence. "No Data Aug" beats baseline at 20 epochs. **Colab-relevant:** with ~10 min window, consider skipping augmentation entirely.

### 4.3 MixUp + CutMix: complementary augmentation

- **MixUp:** linear interpolation of images + labels. Smooths decision boundaries. Excels on small datasets.
- **CutMix:** paste patches between images. Better for tasks needing local features. Maintains accuracy better in later training stages.
- **Best practice:** Combine RandAugment (geometric/color) + MixUp or CutMix (inter-sample). They improve different aspects.

### 4.4 Test-Time Augmentation (TTA): free 1-2% at inference

Run each test sample through N augmented versions, average predictions. No retraining needed. Highly undervalued.

### 4.5 Progressive image resolution: implicit curriculum

Start training at low resolution (e.g., 128px), then increase to target (224px+). Acts like transfer learning from the same dataset. Can reduce total training time 20-40% with no accuracy loss.

---

## 5. Numerical Stability

### 5.1 Log-Sum-Exp: never write softmax + log manually

Always use `F.cross_entropy(logits, targets)` or `F.log_softmax`. The fused kernel subtracts `max(logits)` before exp, preventing overflow in float32 and making float16/bfloat16 training possible at all.

```python
# Wrong: exp(100) = inf → log(inf) = inf → NaN
loss = -torch.log(F.softmax(logits, dim=-1) + 1e-8)

# Correct: CrossEntropyLoss internally uses log_softmax with max subtraction
loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
```

### 5.2 BCEWithLogitsLoss: fused sigmoid + BCE

Use `nn.BCEWithLogitsLoss` — NOT `sigmoid + BCELoss`. The fused version uses the numerically stable formulation: `max(x, 0) - x·t + log(1 + exp(-|x|))`.

### 5.3 FP16 eps=1e-8 rounds to zero

FP16's minimum representable value is ~6×10⁻⁵. Adam's default `eps=1e-8` rounds to zero in FP16 → NaN gradients within 50-200 steps. Fix: `eps=1e-4` or higher for AMP training.

### 5.4 Masked softmax: all tokens masked → NaN

In Transformer attention, masking all tokens with `-inf` then applying softmax produces NaN. Ensure at least one token is visible per row.

---

## 6. Architecture Tricks That Improve Training

### 6.1 `bias=False` before BatchNorm (mandatory)

Linear/Conv + BatchNorm: the bias is immediately subtracted by BN's running mean normalization. It's a dead parameter that wastes compute and can destabilize early training.

### 6.2 Zero-gamma initialization for residual blocks (FixUp/ReZero)

Initialize the last BatchNorm's gamma to 0 in residual blocks. This initializes the block as the identity function, giving dramatically faster early convergence (~2× on ImageNet).

### 6.3 Pre-LN > Post-LN for deep Transformers

Pre-LayerNorm (`x + Sublayer(LN(x))`) provides a gradient highway through residuals. Post-LN (`LN(x + Sublayer(x))`) forces gradients through LN, amplifying variance. Pre-LN trains stably at 12+ layers; Post-LN diverges past ~8 without warmup.

**Already benchmarked:** `projects/nlp/transformer-ln-comparison/`

### 6.4 ResNet-D: avgpool + 1x1 instead of stride-2 1x1

Stride-2 1x1 convolution in residual downsampling silently discards 75% of features. AvgPool + 1x1 preserves all information. +0.5% on ImageNet (Bag of Tricks, He et al. 2019).

### 6.5 Weight init: transformer standard vs PyTorch default

PyTorch Linear defaults to kaiming_uniform (for ReLU nets). Transformers with GeLU/ReLU in FFN + complex attention gradient paths need: `xavier_uniform(gain=1/sqrt(2))` for Linear, `normal(std=d_model^(-0.5))` for embeddings. The difference: **17× lower step-1 gradient norm** (89.0 vs 5.1 on IWSLT2017, d=256).

**Already documented:** `docs/reference/model-gotchas.md` (weight init comparison)

---

## 7. PyTorch Training Loop Gotchas (Silent Bugs)

### 7.1 `model.eval()` alone does NOT disable autograd

`model.eval()` only controls Dropout/BatchNorm behavior. PyTorch still builds the computation graph and stores intermediates. Without `torch.no_grad()`, eval can OOM even when training fits. Always use both:

```python
model.eval()
with torch.no_grad():  # or torch.inference_mode()
    predictions = model(x)
```

### 7.2 `torch.inference_mode()` is faster than `torch.no_grad()`

`torch.inference_mode()` (PyTorch 1.9+) additionally disables version counter bumps and view tracking. Pure inference: always prefer `inference_mode()`.

### 7.3 `zero_grad(set_to_none=True)` saves memory

Default `zero_grad()` writes zeros to gradient tensors. `set_to_none=True` deletes them instead, allowing the allocator to reuse the memory. Saves ~10% of optimizer memory for free.

### 7.4 `retain_graph=True` in training loops → slow memory leak

Never use `retain_graph=True` routinely. It prevents PyTorch from freeing the computation graph after backward. For multiple losses, sum them and call `.backward()` once.

### 7.5 optimizer.zero_grad() BEFORE loss.backward() — or gradients leak across steps

Standard PyTorch pattern:
```python
optimizer.zero_grad(set_to_none=True)  # 1. clear
loss.backward()                         # 2. compute
optimizer.step()                        # 3. update
```

Forgetting step 1 accumulates gradients silently — training looks fine but updates are wrong.

### 7.6 Gradient accumulation: normalize loss correctly

```python
# WRONG: gradients are N× too large (loss.backward() accumulates, not averages)
loss = criterion(output, target)
loss.backward()

# CORRECT: normalize to maintain gradient magnitude
loss = criterion(output, target) / accumulation_steps
loss.backward()
```

### 7.7 `model.train()` / `model.eval()` must toggle per phase

Forgetting to switch modes silently degrades:
- Dropout stays active during eval → noisy predictions, lower accuracy
- BatchNorm uses batch stats during eval → results depend on batch composition
- Dropout off during training → overfitting

### 7.8 `.view()` requires contiguous; `.reshape()` is safer

`view()` fails on non-contiguous tensors (after transpose, permute). `reshape()` copies when needed. Prefer `reshape()` unless you explicitly want a view-only error.

---

## 8. Memory Optimization for T4 (Colab-specific)

### 8.1 AMP cuts memory 40% and speeds up 2×

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    output = model(x)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

T4 Tensor Cores: 65 TFLOPS FP16 vs 8.1 FP32. AMP unlocks them.

### 8.2 Gradient checkpointing: 40-65% memory reduction

`torch.utils.checkpoint.checkpoint()` trades compute for memory. Apply selectively to the most expensive layers (attention, large convolutions). On T4, essential for models >1B params.

### 8.3 Gradient accumulation for batch size > T4 VRAM

Accumulate N micro-batches, optimizer.step() every N. Enables effective batch=256 on T4 where max direct batch=32.

### 8.4 `channels_last` memory format for CNNs

`model = model.to(memory_format=torch.channels_last)`. NHWC layout better utilizes Tensor Cores. 10-30% speedup on conv-heavy models. Not default in PyTorch.

---

## 9. Already-Documented Gotchas (Don't Duplicate)

These are in `docs/reference/model-gotchas.md`:

| Topic | Key finding |
|---|---|
| Noam LR peak trap | Small d_model + short warmup = dangerously high peak LR (6.3× the paper's value) |
| Post-LN vs Pre-LN | Post-LN without warmup → NaN within 100-200 steps |
| Weight init for Transformers | kaiming_uniform → 17× higher step-1 grad norm than xavier_uniform |
| NaN diagnosis flow | Val NaN first, train NaN 1-2 intervals later, Adam v_t contagious |
| Grad clipping band-aid | Clipping 98% of gradient = masking problem, not fixing it |

---

## 10. Colab T4 Recipe: Minimal Overhead, Max Impact

For Colab's ~10 min GPU window, this is the "always start here" config:

```python
# Training config — minimal boilerplate, maximum ROI
model = Model(...)
model = model.to(memory_format=torch.channels_last)  # if CNN
# model = torch.compile(model, mode="reduce-overhead")  # if compatible

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=total_steps,
    pct_start=0.1  # 10% warmup
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda")

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

| Technique | Lines added | Approximate benefit |
|---|---|---|
| AdamW | 0 (replace Adam) | Better generalization |
| OneCycleLR | 1 | 2-10× convergence speedup |
| AMP (autocast) | 3 | 1.5-2× speed, 40% memory |
| Grad clipping | 1 | Prevents NaN |
| Label smoothing | 0 (parameter) | 0.3-1% accuracy |
| `set_to_none=True` | 0 (parameter) | ~10% memory |
| `channels_last` | 1 (if CNN) | 10-30% speedup |

**Total: ~6 lines of additional code for 2-4× effective improvement.**

---

## References

- [A Recipe for Training Neural Networks — Karpathy](http://karpathy.github.io/2019/04/25/recipe/)
- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Bag of Tricks for Image Classification — He et al., CVPR 2019](https://arxiv.org/abs/1812.01187)
- [Super-Convergence — Smith & Topin, 2018](https://arxiv.org/abs/1708.07120)
- [Decoupled Weight Decay / AdamW — Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)
- [Stochastic Weight Averaging — Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)
- [Why Gradients Explode at End of Training — Defazio, Meta FAIR, 2025](https://arxiv.org/abs/2506.02285)
- [Rotational Equilibrium / Weight Decay — Kosson et al., ICML 2024](https://arxiv.org/abs/2305.17212)
- [EleutherAI Cookbook](https://github.com/EleutherAI/cookbook)
- [ml-engineering — stas00](https://github.com/stas00/ml-engineering)
- [HuggingFace: Simple considerations for training LLMs](https://huggingface.co/blog/simple-considerations)
