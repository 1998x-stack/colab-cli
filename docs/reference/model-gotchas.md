# Model Gotchas

Cross-project lessons about model architecture, training behavior, and Colab-specific constraints.

## Training Fundamentals

### Noam LR Scheduler — Peak LR trap: small d_model + short warmup = dangerously high LR

The Noam formula `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))` peaks at `d_model^(-0.5) * warmup^(-0.5)`. With the paper's params (d=512, warmup=4000): peak = 0.00070. With common "small model" params (d=256, warmup=200): peak = 0.00442 — **6.3x higher**.

When scaling down model size, you must scale warmup proportionally to keep the peak LR in a safe range:
- `warmup = k / d_model` where k ≈ 2,048,000 for the paper's peak of 0.0007
- For d=256: warmup ≥ 8000 to match paper's peak
- Quick sanity: `peak = d^(-0.5) * warmup^(-0.5)` — if >0.002, raise warmup

**Observed in:** transformer-ln-comparison (2026-06-14), transformer_iwslt

### Post-LN requires the Noam warmup; Pre-LN is robust without it

Pre-LN's architecture provides a gradient highway through the residual connection (`x + Sublayer(LayerNorm(x))`). Post-LN forces gradients through LayerNorm (`LayerNorm(x + Sublayer(x))`), amplifying variance. This means:

- Post-LN without warmup → NaN within 100-200 steps
- Pre-LN without warmup → trains fine (though warmup still helps convergence speed)
- For fair comparison experiments, use identical hyperparameters — the instability IS the result

**Observed in:** transformer-ln-comparison (2026-06-14)

### Weight Initialization — Default PyTorch init can be 17x worse than the transformer standard

PyTorch Linear defaults to kaiming_uniform (fan-in mode), optimized for ReLU networks. Transformers use GeLU/ReLU in FFN but have complex gradient paths through attention + residuals. The standard transformer init:
- Linear layers: `xavier_uniform` with gain `1/sqrt(2)` (accounts for residual paths)
- Embeddings: `normal(mean=0, std=d_model^(-0.5))`

Step-1 gradient norm comparison (same model, d=256, IWSLT2017):
- kaiming_uniform: grad=89.0
- xavier_uniform + normal embed: grad=5.1 (17x lower)

**Observed in:** transformer-ln-comparison (2026-06-14)

### NaN Diagnosis — Propagation order: val_loss first, train_loss 1-2 intervals later

When weights start producing NaN, it shows up in validation loss first (lower batch count, no gradient clipping protection during eval). Train loss stays finite for 1-2 logging intervals because:
1. Gradient clipping at 1.0 masks the NaN in the backward pass
2. Adam's momentum smooths out individual NaN updates
3. Validation uses larger effective batch (all val data) → more NaN accumulation

**If val_loss=NaN, stop immediately.** The model weights are already corrupted. Continuing just fills your metrics CSV with NaN rows.

### NaN is contagious through Adam's second moment

Once a single weight becomes NaN, Adam's `v_t` (running variance) accumulates NaN. Within 1-2 steps, all weights propagate NaN through the exponential moving average. You cannot recover — must restart from the last clean checkpoint.

### Gradient Clipping — Clipping at 1.0 can mask but not prevent NaN

With unclipped gradients of ~50-120 (typical post-LN early training), clipping to 1.0 cuts 98%+ of the gradient magnitude. This keeps training alive temporarily but doesn't fix the underlying instability. The clipped gradients are still large enough to push weights into regimes where the next forward pass produces NaN activations.

**Fix the LR schedule and initialization** — don't rely on aggressive clipping as a band-aid.

---

## Project-Specific Gotchas

Field-tested issues encountered when running ML models on Colab VMs.

## nanoGPT (karpathy/nanoGPT)

Date: 2026-06-10 | GPU: T4 | Free tier

### 1. `configure_optimizers` missing from stripped-down GPT class

When copying the GPT model from `model.py` into a self-contained training script, the `configure_optimizers()`, `get_num_params()`, and `estimate_mfu()` methods are easy to miss. The original train.py calls `model.configure_optimizers(...)` — without it you get `AttributeError: 'GPT' object has no attribute 'configure_optimizers'`.

- `configure_optimizers` also requires `import inspect` for the fused AdamW check.
- `estimate_mfu` requires `get_num_params`.
- Copy all three methods if you're bundling the model into a single script.

### 2. `torch.cuda.amp.GradScaler` is deprecated in PyTorch 2.11

Colab T4 VMs ship PyTorch 2.11.0+cu128. The old API raises a `FutureWarning`:

```python
# Broken (deprecated):
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# Fixed:
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
```

### 3. T4 does not support bfloat16 compilation natively

`torch.compile(model)` with bfloat16 autocast emits:
```
UserWarning: Tesla T4 does not support bfloat16 compilation natively, skipping
```
Training still works (falls back to eager bfloat16 matmul), but compilation speedup is lost. The model runs in eager mode with bfloat16 autocast — slower than expected but functionally correct. Consider `dtype='float16'` if compilation matters, or skip `torch.compile` on T4.

### 4. `estimate_loss()` returns tensors, not Python floats

`torch.zeros(...).mean()` returns a 0-d tensor, not a Python float. When saving to JSON:

```python
# Broken:
out[split] = losses.mean()          # tensor(1.23)
# Fixed:
out[split] = losses.mean().item()   # 1.23
```

Without `.item()`, `json.dump(metrics)` crashes with `TypeError: Object of type Tensor is not JSON serializable` after the entire training loop completes — wasting the run. The metrics file gets truncated mid-write.

### 5. Free-tier T4 sessions die in ~10-12 minutes

With the Shakespeare char config (10.75M params, batch_size=64, block_size=256), 500 iterations takes ~7 minutes. 800 iterations (~10.5 min) died 2 minutes before finishing on two separate attempts. Budget ~7 minutes of actual training time — if it takes longer, reduce `max_iters` or use a smaller model.

### 6. `no_proxy` is required for `colab exec`/`colab download` (China)

REST API calls (`colab new`, `colab stop`) work through the SOCKS5 proxy, but WebSocket kernel connections and session file downloads often fail with `SSLError: UNEXPECTED_EOF_WHILE_READING` when routed through `ALL_PROXY=socks5://127.0.0.1:7890`. Set `no_proxy` before exec/download:

```bash
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

This bypass is session-specific — sometimes it works without, sometimes it doesn't. Retry with 2-3 attempts when downloads fail.

### 7. Checkpoint size bloats with torch.compile

With `torch.compile`, the checkpoint (`ckpt.pt`) is ~126 MB vs ~42 MB without compilation (for the 10.75M param model). This is because `torch.compile` stores compilation artifacts in the state dict. For checkpoint-only workflows where compilation artifacts don't need to be preserved, skip compile or extract only the model weights.

---

## nanochat (karpathy/nanochat)

Date: 2026-06-10 | GPU: T4 | Free tier | depth=6 (73.5M params)

### 1. Auto-computed batch size kills T4 training speed

nanochat auto-computes `total_batch_size` from scaling laws (262K tokens for d=6). With `--device-batch-size=1 --max-seq-len=256`, each micro-batch is only 256 tokens → 1024 gradient accumulation steps → ~77s per optimizer step → 500 iterations would take ~10.7 hours. **Always explicitly set `--total-batch-size` on T4:**

```bash
# Broken (auto-computed): grad_accum_steps = 262144/256 = 1024
python -m scripts.base_train --depth=6 --device-batch-size=1

# Fixed: grad_accum_steps = 16384/256 = 64
python -m scripts.base_train --depth=6 --device-batch-size=1 --total-batch-size=16384
```

With `--total-batch-size=16384`, each step takes ~900ms on T4, and 250 iterations complete in ~4 minutes.

### 2. `--window-pattern=L` required for SDPA fallback on T4

T4 (SM 7.5) doesn't support Flash Attention 3. nanochat falls back to PyTorch SDPA, which has **no support for sliding window attention patterns** (`SSSL`, etc.). Using the default window pattern without FA3 produces incorrect attention and warnings:

```bash
python -m scripts.base_train --window-pattern=L  # Full attention only on T4
```

### 3. `NANOCHAT_DTYPE=float16` — T4 has no bf16 support

T4 SM 7.5 < 8.0, so nanochat auto-detects `float32` as compute dtype. This disables tensor cores entirely. Override with float16:

```bash
NANOCHAT_DTYPE=float16 python -m scripts.base_train ...  # Enables fp16 tensor cores + GradScaler
```

Without this, training runs in float32 at ~3x slower throughput. The GradScaler is automatically created by base_train.py when `COMPUTE_DTYPE == float16`.

### 4. Checkpoint bloat kills proxy downloads

nanochat saves optimizer states alongside model weights in checkpoints. For d=6 (73.5M params), the final checkpoint is ~700MB. SOCKS5 proxy downloads reliably fail around 125MB with `IncompleteRead`. **Skip checkpoints in output tarballs:**

```python
# Only package log + plots + tokenizer (497 KB total, reliable download)
with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
    tar.add("/content/train.log", arcname="train.log")
    tar.add("/content/plots", arcname="plots")
    # Skip /content/nanochat-data/base_checkpoints/ — 700MB+
```

Total output tarball: <500 KB without checkpoints vs 744 MB with them.

### 5. `colab run` auto-terminates — can't download artifacts

`colab run --gpu T4` destroys the VM immediately after the script exits. All files vanish. Use the persistent workflow instead:

```bash
cb new --gpu T4 -s mysession           # 1. Provision (VM persists)
cb exec -f script.py --timeout 900     # 2. Run training
cb download /content/output.tar.gz ./  # 3. Download artifacts
cb stop -s mysession                   # 4. Clean up
```

Chain `exec` && `download` && `stop` in a single command to minimize the window between training completion and session death.

### 6. `uv sync` installs a separate torch even though Colab has one

Colab VMs have PyTorch 2.11.0 pre-installed, but nanochat pins `torch==2.9.1` in `pyproject.toml`. `uv sync --extra gpu` downloads a fresh torch (~2GB) into `.venv`, adding ~85s to setup. The system torch is unused. If you're extending nanochat for Colab, consider relaxing the torch version pin and using system torch to save setup time.

### 7. MFU reports 0.00% on T4 with fp16

The MFU calculation uses bfloat16 peak FLOPS as the reference, but T4 runs in float16 mode. The reported `bf16_mfu` field is `0.00` throughout training — not a bug, just the wrong reference. Use `tok/sec` for throughput comparisons on T4 (~17,000 tok/sec achieved for d=6 with device-batch-size=1).

### 8. Session death during exec proxy hiccups

Even with a healthy detached training process running on the VM, the session can get pruned after SSL/proxy errors on the `colab exec` WebSocket connection. Two defensive patterns:

```bash
# A. no_proxy for WebSocket domains (try first)
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

# B. Chain exec + download so download happens immediately if exec succeeds
cb exec -f run_all.py --timeout 900 && cb download /content/out.tar.gz ./
```

Retry 2-3 times on SSL errors — they're often transient. Check `cb sessions` after failures to see if the VM is still alive.

---

## Transformer IWSLT (Attention Is All You Need)

Date: 2026-06-11 | GPU: T4 | Free tier | 61M params | 206K pairs

### 1. IWSLT 2017 data access: 5 failed approaches before finding the right one

The IWSLT 2017 De-En dataset is surprisingly hard to download reliably on Colab:

| Attempt | Approach | Error |
|---------|----------|-------|
| 1 | Direct URL `wit3.fbk.eu/.../de-en.tgz` | Returns HTML login page (Google auth wall), not gzip |
| 2 | `datasets.load_dataset("iwslt2017", ...)` | Colab's datasets too new — "Dataset scripts are no longer supported" |
| 3 | `datasets==2.14.0` pinned | No `trust_remote_code` support, builder config error |
| 4 | HF CDN raw `.../iwslt2017/resolve/main/data/de-en/train.de` | Lowercase org `iwslt2017` → 307 redirect (urllib doesn't follow 307) |
| 5 | HF CDN raw `.../IWSLT/iwslt2017/.../de-en.zip` | 404 — wrong path in repo |

**Final working approach:** Use canonical uppercase org + correct ZIP path + urllib:
```python
# IWSLT → 302 redirect (urllib follows). iwslt2017 → 307 (urllib doesn't).
url = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/de/en/de-en.zip"
urllib.request.urlretrieve(url, zip_path)
# ZIP contains de-en/train.tags.de-en.{de,en} — plain text after XML meta tags
```

### 2. IWSLT training files: plain text, not `<seg>` wrapped

The `train.tags.de-en.{de,en}` files have XML meta tags (`<doc>`, `<url>`, `<speaker>`, `<talkid>`, `<title>`, `<description>`) in the header, followed by plain text sentences (one per line). They do NOT use `<seg>` tags — unlike what the dataset script suggests. Parser must:
- Skip lines starting with `<` (meta tags)
- Treat everything else as sentence pairs
- Filter in parallel (both DE and EN lines together, not independently)

### 3. DataLoader num_workers>0 hangs on Colab

With `num_workers=2` and the Rust-backed `tokenizers` library, `DataLoader` stalls silently after model init. No error, no crash — just no forward progress. `num_workers=0` fixes it. Pre-tokenization eliminates any throughput concern.

### 4. Training appears "stuck at Params" but is actually running

The log prints "Params: 61,009,920" then nothing for 5+ minutes. Training IS running — CUDA JIT compilation happens on the first batch (2-3 min), then the first epoch is 4-5 min. Per-epoch logging means no intermediate output. Check `nvidia-smi` or `ps aux` to confirm.

### 5. First epoch overhead: 7-10 min before first log line

Breakdown on fresh VM:
- ZIP download: 30s
- ZIP extraction: 10s
- BPE tokenizer training (32K vocab, 206K pairs × 2): 60s
- Pre-tokenization (optional but essential): 30s
- Model init 61M params to GPU: 10s
- CUDA JIT compilation (first batch): 2-3 min
- First epoch training: 2-5 min

Total ~7-10 min. With ~12-15 min WebSocket window, first epoch barely fits. Second session onward (all data cached, no JIT) gets 3-4 epochs.

### 6. AMP on T4: 2-3× speedup

```python
# PyTorch 2.11 API (not torch.cuda.amp — deprecated)
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

T4 tensor cores give ~8× matmul throughput in float16 vs float32. Per-epoch drops from ~5 min to ~2 min.

### 7. Pre-tokenization eliminates 80% of DataLoader overhead

Per-sample `tokenizer.encode()` in `__getitem__` is the dominant bottleneck. With 206K pairs, that's 206K encode calls per epoch. Pre-tokenize once after training the tokenizer, save to `.pt`, reload instantly:

```python
# One-time after tokenizer training
data = [(torch.tensor(encode(de)), torch.tensor(encode(en))) for de, en in pairs]
torch.save(data, "/content/iwslt_data/train.pt")

# Dataset loads pre-tokenized tensors directly
class TranslationDataset(Dataset):
    def __init__(self, ...):
        self.data = torch.load(cached_path, weights_only=False)
    def __getitem__(self, idx):
        return self.data[idx]  # instant
```

### 8. Beam search finished-beam handling

When a beam produces EOS, it must only allow EOS on subsequent steps (force `log_probs[finished, :] = -inf; log_probs[finished, eos_idx] = 0`). Otherwise the beam keeps generating tokens after EOS, wasting compute and diluting scores.

### 9. BLEU evaluation is the hidden bottleneck (hours per epoch)

BLEU eval runs beam search on the entire validation set. With 41K validation pairs × beam_size=4 × max_len=128, each epoch's BLEU eval takes ~3-5 HOURS. The training appears stuck after the first epoch because BLEU never completes.

**Fix:** Use a tiny validation subset (100 sentences) with greedy decode (beam_size=1) during training. This takes ~10s instead of hours. Save proper BLEU with full beam search for final evaluation only.

```python
val_bleu_ds = TranslationDataset(val_pairs[:100], tokenizer, max_len, name="val_bleu")
bleu = evaluate(model, val_bleu_loader, tokenizer, device, beam_size=1, max_len=96)
```

### 10. CUDA OOM during eval — model fits training but not evaluation

T4 has 14.56 GiB VRAM. The model uses ~9.3 GiB during training with batch_size=32. But during BLEU evaluation, beam search creates additional tensors that cause OOM. Also, CUDA memory fragments over time.

**Fixes:**
- `batch_size=32` (not 64) for training
- `torch.cuda.empty_cache()` before val_loss and BLEU eval
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation
- Use beam_size=1 during training eval (smaller memory footprint)

### 11. `flush=True` is essential for subprocess log files

Even with `PYTHONUNBUFFERED=1` and `python -u`, print() output redirected to a file via `subprocess.Popen` can be delayed by minutes. Training appears "stuck at Params: 61,009,920" while actually running (GPU at 77% utilization).

**Fix:** Add `flush=True` to every print:
```python
print(f"[train] Params: {n:,}", flush=True)
print(f"[train] Epoch {e}: loss={l:.3f}", flush=True)
# Progress every 200 batches
if n_batches % 200 == 0:
    print(f"  batch {n_batches}: loss={loss:.3f}", flush=True)
```

**Verification:** Don't trust log timestamps. Check `nvidia-smi` GPU utilization and `ps aux` CPU time — if GPU >50% and process has been running for minutes, training IS happening.

### 12. Checkpoint download fails above ~600MB through China proxy

Full checkpoint (model weights + Adam optimizer moments) = ~1GB. Proxy connection reliably breaks at ~624MB with `IncompleteRead`. Gzip compression at level 3 only reduces to ~500MB — still too large.

**Fix:** Save TWO checkpoints per epoch:
1. **Full checkpoint** (`checkpoint_epochN.pt`, gzip, ~500MB) — for training resume on the VM
2. **Weights-only checkpoint** (`weights_epochN.pt`, ~120-233MB) — for proxy download

```python
# Full (VM-local resume): model + optimizer + scheduler state
save_checkpoint(ckpt_path, model, optimizer, scheduler, ...)

# Weights-only (proxy download): model weights + metrics
save_weights(w_path, model, epoch, metrics, config)
```

Weights-only is ~233MB for 61M params — downloads reliably through proxy. Full checkpoint is 500MB (gzip) or 1GB (raw).

### 13. Smoke test: verify full pipeline locally before Colab

A 200-pair, 3-epoch smoke test on Colab verifies the entire pipeline in 96 seconds:
- AMP mixed precision works
- Beam search works
- Checkpoint save/load works
- BLEU evaluation works
- CUDA OOM doesn't happen at small scale

Run this BEFORE deploying real training — catches 80% of bugs in 2 minutes instead of wasting session cycles.

### 14. Pre-tokenization cache key collision

When multiple `TranslationDataset` instances (train, val, val_bleu) share the same cache path, the val set loads training data. Fix: key each dataset by name.

```python
# Broken: all datasets share "train.pt"
cached = os.path.join(data_dir, "train.pt")

# Fixed: each dataset has its own cache
def __init__(self, ..., name="train"):
    cached = os.path.join(data_dir, f"{name}.pt")
```

---

## CUDA Dark Corners — cross-project findings

Verified on Colab T4 (CUDA 12.8, PyTorch 2.11.0+cu128). 19 experiments across 8 categories. See `projects/systems/cuda-dark-corners/` for full results.

### cross_entropy + permute is 17-118× slower for LLM logits

`F.cross_entropy(logits.permute(0, 2, 1), targets)` converts LLM-standard (B, S, V) layout to cross_entropy's (N, C) expectation. The internal permute triggers slow reduction kernels.

At B=1, S=128, V=50257:
- CE+permute: **51.6 ms**
- log_softmax+gather: **0.45 ms** (114.7× faster)

The fix:
```python
# ❌ 51.6 ms — permute triggers slow kernel
loss = F.cross_entropy(logits.permute(0, 2, 1), targets)

# ✅ 0.45 ms — no layout change needed
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
```

`.contiguous()` after permute does NOT help — the bottleneck is the reduction kernel path, not memory contiguity. Smaller batches are affected worse (fixed overhead dominates).

**Observed in:** cuda-dark-corners/layout-002 (2026-06-14). Affects ALL LLM training code.

### CUDA 12.8 / PyTorch 2.11 fixed multiple "classic" traps

The following well-documented traps were **NOT observed** on this stack. Old optimization advice referencing them may be obsolete:

| Trap (source) | Old expectation | Actual (CUDA 12.8) | Root cause |
|---------------|----------------|--------------------|------------|
| Implicit `.contiguous()` copies on non-contiguous op chains | 2-10× slowdown, 5-15 copies | 1.0-1.1×, 1 copy | stride-aware kernels improved |
| `index_select` 2-6× slower for 2D+ | 2-6× | 1.0-1.2× | index_select gather kernel optimized |
| FP16 eps=1e-8 NaN within 50-200 steps | NaN in <200 steps | No NaN in 500 steps | AMP `GradScaler` dynamic scaling prevents underflow |
| CUDA first-call tax ~1.6s | 1.6s | ~389ms | CUDA 12.8 init path optimization |
| Ad-hoc `.pin_memory()` 1.5-2× slower | 1.5-2× | 0.6-1.2× | CUDA 12.8 internal pinned staging buffer |
| `torch.compile` 8× worse on non-contiguous `max()` | 8× | **1.0×** (eliminates penalty) | inductor 3-stage→2-stage reduction |

**Implication:** Always verify "known" performance traps on your target CUDA/PyTorch version. Auto-upgrades may have already fixed them.

**Observed in:** cuda-dark-corners (layout-001, layout-003, precision-001, launch-002, transfer-003, compile-002) — 2026-06-14.

### CUDA timing: perf_counter without synchronize() is 3-15× wrong

`time.perf_counter()` without `torch.cuda.synchronize()` measures CPU kernel launch latency (~15-70µs), NOT GPU execution time. The error grows with kernel duration:

| Matmul | No sync (µs) | Actual GPU (CUDA events) | **Underestimate** |
|--------|-------------|--------------------------|-------------------|
| 64×64 | 27 | 83 | 3.1× |
| 256×256 | 27 | 207 | **7.7×** |
| 1024×1024 | 71 | 525 | **7.4×** |

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

**Observed in:** cuda-dark-corners/sync-003 (2026-06-14). Affects ALL GPU benchmarking code.

### N(0, 1.0) init causes catastrophic divergence

Initializing a simple MLP with `N(0, 1.0)` produces initial loss of 2051 (expected: 2.30 for K=10). Kaiming uniform and Xavier uniform both give ~2.31-2.34.

Always run the "loss at init" sanity check before full training:
```python
# K-class softmax: initial loss should be -ln(1/K)
# K=10 → 2.3026. If > 3.0, init or loss function is broken.
loss = F.cross_entropy(model(randn_input), randn_labels)
assert abs(loss.item() - 2.3026) < 0.5, f"Bad init: loss={loss.item():.2f}"
```

**Observed in:** dl-training/dltrain-002 (2026-06-14). Karpathy Recipe #2.

### Overfit single batch is the most efficient debug check

Before training on full dataset, verify the model can memorize 16 fixed samples to near-zero loss (<0.01 in 200 steps). Remove dropout, weight decay, and augmentation for this test. Catches ~80% of structural bugs.

**Observed in:** dl-training/dltrain-001 (2026-06-14). Karpathy Recipe #1.

### view() crashes on permuted tensors; reshape() doesn't

After `.permute()`, `.transpose()`, or `.T`, tensors become non-contiguous. `.view()` requires contiguous memory and crashes with `RuntimeError`. `.reshape()` copies internally when needed — always safe.

```python
x = x.permute(0, 2, 1)
x = x.view(-1)      # ❌ RuntimeError
x = x.reshape(-1)   # ✅ works (copies if needed)
```

**Observed in:** dl-training/dltrain-011 (2026-06-14). Karpathy's 6 common mistakes. Particularly dangerous in transformer code with frequent permute/transpose.

### T4 Tensor Cores utilization peaks at 37%

T4's 65 TFLOPS FP16 theoretical peak is unreachable in practice for single matmul. Peak observed: 23.8 TFLOPS at 3072×3072 (37% utilization). FP16 is still 6.4× faster than FP32 at 8192×8192. Tensor Cores start activating around 384×384.

For maximum throughput on T4: target matmul dimensions ≥768, use FP16 (not BF16 — SM 7.5 doesn't support), and avoid mixed small/large matmuls in the same model.

**Observed in:** cuda-dark-corners/precision-002 (2026-06-14).

### GPU is NOT always faster — know the crossover

On T4:
- **Matmul**: GPU starts winning at ~128×128. At 64×64, CPU is faster.
- **Element-wise ops** (relu, add): GPU only wins above ~100K elements. Below that, kernel launch overhead dominates.

For small-tensor workloads (token-level operations, small batch linear layers), CPU may be faster. Always benchmark before assuming GPU > CPU.

**Observed in:** cuda-dark-corners/launch-001 (2026-06-14).
