# Model Gotchas

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
