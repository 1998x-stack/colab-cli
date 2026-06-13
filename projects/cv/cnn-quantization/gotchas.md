# CNN Quantization — Gotchas & Report

Field-tested on Colab T4, 2026-06-13. 5 sessions across 4 accounts.

## Quantization-specific

### 1. `torch.ao.quantize_dynamic` is CPU-only
**Symptom:** `NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend.`  
**Why:** Colab's PyTorch build doesn't include CUDA quantized op kernels. `quantized::linear_dynamic` only has CPU, Meta, and autograd backends.  
**Fix:** Keep INT8 model on CPU. Use `model_device = next(model.parameters()).device` in eval/latency functions to auto-detect.  
**Tradeoff:** INT8 latency measured on CPU (slower), not directly comparable to FP32/FP16 CUDA latency.

### 2. FP16 eval needs input dtype matching
**Symptom:** `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`  
**Why:** Model is cast to `.half()` but DataLoader still produces float32 tensors.  
**Fix:** Detect model dtype in evaluate/measure_latency: `model_dtype = next(model.parameters()).dtype` then `imgs.to(device, dtype=model_dtype)`.

### 3. HF `datasets` CIFAR-10 images are nested lists
**Symptom:** `TypeError: Cannot handle this data type: (1, 1, 32, 3), |u1`  
**Why:** `uoft-cs/cifar10` stores images as nested lists. `np.array(img)` produces 4D tensors from batch wrapping.  
**Fix:** Use `torchvision.datasets.CIFAR10` instead. Same data, reliable PIL Image output. HF token not needed for CIFAR-10 (public dataset).

### 4. INT4 state dict loading needs `strict=False`
**Symptom:** `Missing key(s): ... Unexpected key(s): ...`  
**Why:** `Int4QuantizedLinear` replaces nn.Linear → state dict keys change (`weight` → `q_weight` + `scales`). Loading FP32 state dict into quantized model fails with strict matching.  
**Fix:** `int4_model.load_state_dict({k: v for k, v in fp32_state.items() if k in int4_model.state_dict()}, strict=False)` — or quantize before loading weights.

### 5. `colab exec` env vars don't propagate to VM
**Symptom:** `LAUNCH_DEPS` env var set locally but ignored on VM.  
**Why:** `colab exec -f` sends the file to VM kernel — local shell env vars are NOT visible to the remote Python process.  
**Fix:** Hardcode defaults in launch.py. For runtime overrides, use stdin pipe to set `os.environ` before import.

## Colab infrastructure

### 6. CIFAR-10 download speed varies 10×
**Timings across 5 sessions:** 13s (12.5 MB/s) to 3m41s (~0.7 MB/s).  
**Why:** `cs.toronto.edu` is not on Google CDN. Network contention on Colab's shared uplink.  
**Mitigation:** Warmup session downloads data → session dies → re-provision and train with cached `/content/data/`. Or use `--skip_train` for quantization-only reruns on the same session.

### 7. 10-min GPU window is real but variable
**Observed:** Sessions die at 4–10 min, not exactly 10 min.  
**Pattern:** Download-heavy sessions die sooner (network activity counts toward the window).  
**Practical limit:** Budget 8 min for GPU compute after all downloads complete.

### 8. WebSocket exec drops are independent of session health
**Symptom:** `RuntimeError: Connection was lost.` mid-exec but `colab sessions` shows session alive.  
**Why:** WebSocket path (through GFW proxy) drops while REST keep-alive path stays up.  
**Workflow:** After any exec drop, verify session via `colab sessions` (REST). Download tar via `colab download` (REST). Only re-provision if 404.

## Design notes

### 9. cleanrl-style log function: global is better than closure
**Symptom:** `ValueError: I/O operation on closed file` when log function captures a `with` block file handle.  
**Why:** Python closures capture the file object, not the open/closed state. When the `with` block exits, the handle closes but the closure still references it.  
**Fix:** Module-level `log()` that does `open(path, "a")` each call. Simple, crash-safe, works across all functions.

### 10. CIFAR-10 10k subset is enough for quantization comparison
**Observation:** 72% val acc on 10k train / 2k val is sufficient to compare quantization methods. Full 50k would take 5× longer (~7 min) and push past the 10-min window.  
**Recommendation:** For quantization comparison projects, use the smallest dataset that gives a non-trivial baseline (>random + 2×). 72% is adequate — the goal is measuring quantization error, not SOTA accuracy.
