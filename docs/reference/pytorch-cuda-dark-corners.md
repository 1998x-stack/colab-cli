# PyTorch & CUDA Performance Dark Corners — A Catalog for T4 Benchmarking

Date: 2026-06-14 | GPU: Tesla T4 | PyTorch 2.11.0+cu128 | CUDA 12.8

Research synthesis from PyTorch GitHub issues, CUDA documentation, web research, and our own Colab experiments. Each entry describes a non-obvious performance behavior, why it's surprising, expected magnitude, and whether it can be benchmarked on a T4 with a simple script.

---

## 1. CPU-GPU Transfer

### 1.1 GPU→CPU transfer is 9× slower without pinned memory

**The surprise:** `.to()` looks symmetric but isn't. GPU→CPU runs at 1.4 GB/s vs CPU→GPU at 6.9 GB/s on T4. With pinned memory or `non_blocking=True`, both hit ~13.3 GB/s.

**Root cause:** CUDA DMA can't target pageable memory. Default `.to('cpu')` triggers an invisible alloc+DMA+memcpy+free pipeline where the CPU-driven memcpy is the bottleneck.

**Already benchmarked and documented:** `docs/reference/pytorch-cpu-gpu-transfer.md` + `projects/systems/pytorch-transfer-benchmark/`

**T4 benchmark:** Done. See above.

### 1.2 Combined `.to(device, dtype)` is 3-8× slower than two-step

**The surprise:**
```python
# Slow (5-8x): casts uint8→float32 on CPU (4x data → 4x PCIe time), then transfers
x_gpu = img.to(device="cuda", dtype=torch.float32)

# Fast: transfers small uint8 to GPU first, then casts where bandwidth is free
x_gpu = img.to("cuda").to(torch.float32)
```

The API encourages the combined one-liner, but the ordering is wrong: it casts on CPU first (bloating data size), then transfers the larger payload. The two-step approach transfers the compact dtype first, then does the cheap GPU-side cast.

**Why surprising:** The "obvious" single call is up to 8× slower. This affects every image pipeline that loads uint8 images from disk.

**T4 benchmark:** Compare combined vs two-step for uint8 tensors of varying sizes (256×256, 512×512, 1024×1024). Measure wall time.

**Expected magnitude:** 3-8× speedup for two-step on large images.

### 1.3 Ad-hoc `.pin_memory()` before `.to()` is counterproductive

**The surprise:** `tensor.pin_memory().to("cuda", non_blocking=True)` is often slower than just `tensor.to("cuda")` because CUDA already creates a pinned staging buffer internally. You're doing the pinning work twice — once for the explicit `pin_memory()` call (cudaHostAlloc), and once for the internal staging buffer. Pin in DataLoader, not ad-hoc.

**T4 benchmark:** Compare `.to("cuda")` vs `.pin_memory().to("cuda", non_blocking=True)` for tensors of various sizes.

**Expected magnitude:** 1.5-2× slower for the "optimized" pattern.

---

## 2. Kernel Launch & GPU-vs-CPU Crossover

### 2.1 GPU is slower than CPU for small tensors

**The surprise:** The GPU is not always faster. There is a clear crossover point below which CPU wins:

| Operation | CPU faster below | Why |
|---|---|---|
| Element-wise (add/mul/sin) | ~10,000 elements | 5–15 µs kernel launch overhead dominates |
| Matrix multiply (FP32) | ~100×100 to 256×256 | Below this, CPU blas wins on dispatch cost |
| Batched tiny matmuls | Single-at-a-time is 1,841× slower than fused batch | Each launch costs as much as the entire fused op |

Even BERT-Base at batch=1 on A100 is slower than on V100 because framework overhead (not GPU compute) dominates latency at small batch sizes. This worsens with newer, faster GPUs — the GPU gets faster but the launch overhead stays constant.

**Why surprising:** The "GPU = fast" assumption is deeply ingrained. The crossover point is much larger than most people guess — 100×100 for matmul means most "small matrix operations" people casually push to GPU would be faster on CPU.

**T4 benchmark:** Sweep matmul sizes from 1×1 to 1000×1000 on both CPU and GPU. Plot throughput vs size. Find the empirical crossover point. Same for element-wise ops.

**Expected magnitude:** 2-100× faster on CPU for small tensors. Crossover at ~100×100 (matmul), ~10K elements (element-wise).

### 2.2 CUDA first-call tax: 1.6 seconds for the first operation

**The surprise:** `torch.tensor([1]).cuda()` takes ~1.6 seconds on first call and ~30 µs on subsequent calls — a 50,000× difference. This includes CUDA context creation, PTX compilation, kernel loading, and cuBLAS/cuDNN lazy initialization.

`torch.cuda.init()` does NOT fully initialize CUDA — it only allocates ~3 MB. Full context init only happens on the first kernel launch or tensor creation. Any benchmark that doesn't discard the first iteration is measuring initialization overhead, not computation.

**Why surprising:** The first CUDA call in any Colab session secretly costs 1.6 seconds. No warning, no progress bar.

**T4 benchmark:** Time first CUDA call vs 10 subsequent calls. Show that warmup is mandatory for realistic benchmarks. Also measure what `torch.cuda.init()` actually covers vs what the first `torch.tensor().cuda()` covers.

**Expected magnitude:** ~1.6s first call, ~30 µs after (50,000×).

---

## 3. Hidden Synchronization Points

### 3.1 `.item()` secretly calls `cudaStreamSynchronize()`

**The surprise:** Every `loss.item()` or `acc.item()` in a training loop forces a full GPU pipeline flush. Computing all metrics as GPU tensors and doing a single `.tolist()` reduces N syncs to 1 sync. At 50K steps with 4 metrics, per-metric `.item()` adds 27% to total wall time (3.2 hrs vs 2.5 hrs).

`.item()` looks like a trivial scalar read but serializes the entire GPU pipeline. Each call blocks until all previously queued kernels complete.

**Why surprising:** A one-character change (`.item()` vs `.tolist()`) has a 10-30% training throughput impact.

**T4 benchmark:** Training loop with per-metric `.item()` vs batched `.tolist()` for 1,000 steps. Vary number of metrics (1, 4, 10).

**Expected magnitude:** 10-30% slowdown from per-metric syncs.

### 3.2 Adding `cuda.synchronize()` can make training 50% FASTER

**The surprise:** On GTX 1080 with LSTM + CrossEntropyLoss + AdamW, adding `torch.cuda.synchronize()` between `backward()` and `step()` made training 50% faster. This contradicts the normal rule that sync = slowdown.

The mechanism: without sync, the CUDA stream fills with queued ops, and the optimizer step launches on GPU memory still being used by backward — triggering implicit stream dependencies and serialization. The explicit sync flushes the stream, letting the optimizer start clean.

**Why surprising:** Sync = faster. The exact opposite of conventional wisdom.

**T4 benchmark:** Train an LSTM with and without sync between backward/step. This is version and GPU-specific — may not reproduce on T4/CUDA 12.8 but worth testing.

**Expected magnitude:** 0-50% depending on model and GPU.

### 3.3 Dynamic shapes (boolean masking, nonzero) force hidden syncs

**The surprise:** Boolean indexing with a GPU mask forces PyTorch to transfer data back to CPU to determine output size — because the number of True elements in a mask is unknowable without computing it.

```python
x[gpu_mask]          # forces CPU sync to determine output shape
torch.nonzero(x)     # same — output size depends on values
torch.where(cond, a, b)  # fixed output size — no sync
```

`torch.where` can be 2-5× faster than boolean indexing for sparse masks because it avoids the hidden sync.

**T4 benchmark:** Compare `torch.where(cond, a, b)` vs `a[cond]` for varying mask sparsity (1%, 10%, 50%, 90%) on tensors of different sizes.

**Expected magnitude:** 2-5× faster with `torch.where` for sparse masks.

### 3.4 CUDA timing without synchronize is 10-100× wrong

**The surprise:** Without `torch.cuda.synchronize()`, `time.perf_counter()` measures CPU submission time (microseconds), not GPU execution time (milliseconds). Many self-written benchmarks silently measure the wrong thing. PyTorch's own operator benchmark framework had this bug for backward pass timing until 2026.

```python
# Wrong — measures CPU launch time (~5 µs)
t0 = time.perf_counter()
result = torch.matmul(a, b)
elapsed = time.perf_counter() - t0  # 5 µs, actual GPU time was 500 µs

# Correct — measures GPU execution time
t0 = time.perf_counter()
result = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0  # 505 µs
```

**T4 benchmark:** Time a large matmul with and without `torch.cuda.synchronize()`. The unsynchronized time can be 10-100× smaller.

**Expected magnitude:** 10-100× timing discrepancy without sync.

---

## 4. Tensor Layout & Memory Access

### 4.1 Implicit `.contiguous()` copies chain silently

**The surprise:** After `.T`, `.permute()`, or strided slicing, many operations silently call `.contiguous()` — triggering a full data copy each time. A chain of 10 operations (matmul + layer_norm + add...) on a transposed tensor can copy the data 5-15 times, each time allocating new memory.

```python
x = x.T                    # free — just metadata
x = x + bias               # triggers .contiguous(), copy #1
x = F.layer_norm(x, ...)   # triggers .contiguous(), copy #2
x = F.relu(x)              # OK — relu works on non-contiguous
x = torch.matmul(x, w)     # triggers .contiguous(), copy #3
```

**Why surprising:** Permutation is "free" (just strides), but every downstream op pays the cost repeatedly. The user never sees these copies — they happen inside PyTorch dispatch.

**T4 benchmark:** Chain 10 operations on contiguous vs transposed tensor. Use `torch.autograd.profiler` to count `aten::copy_` calls. Measure total wall time.

**Expected magnitude:** 2-10× slowdown on non-contiguous tensor chains.

### 4.2 `cross_entropy` is 29× slower than `log_softmax+gather` for LLM logits

**The surprise:** On A100 with LLM logit shapes (batch=2, seq_len=512, vocab=50257):

```python
# 76.8 ms
loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction='none')

# 2.65 ms — same numerical result
loss = F.log_softmax(logits, dim=-1)
loss = -loss.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
```

The `permute` inside `cross_entropy` creates a non-contiguous tensor, and reduction on non-contiguous tensors takes an entirely different (slower) kernel path.

**Why surprising:** The "fused function is faster" intuition is reversed. `cross_entropy` expects channels in dim=1, but LLM logits have vocab in dim=-1 — forcing a permute that destroys the fusion benefit.

**T4 benchmark:** Time both approaches with LLM-typical shapes. The ratio on T4 will differ from A100 but the pattern should hold.

**Expected magnitude:** 10-30× slowdown for `cross_entropy` with mismatched dim layout.

### 4.3 `index_select` is slower than regular indexing for 2D+ tensors

**The surprise:** `torch.index_select` has an optimized fast path only for 1D tensors. For 2D+ tensors, `x[:, idx]` is 2-6× faster because it broadcasts indices and uses pointwise operations instead of launching a specialized gather kernel.

```python
# Slower for 2D tensors
torch.index_select(x, dim=1, index=idx)

# Faster for 2D+ (2-6×)
x[:, idx]
```

**Why surprising:** The dedicated, named function is slower than the generic syntax for the most common case.

**T4 benchmark:** Compare `torch.index_select` vs `x[:, idx]` for 1D, 2D, 3D tensors.

**Expected magnitude:** 2-6× faster with regular indexing for 2D+.

---

## 5. Memory Allocator

### 5.1 Allocation order determines whether you OOM

**The surprise:** PyTorch's caching allocator manages memory in fixed-size segments (2 MiB for small, 20 MiB for large). Allocating small-then-large can cause 2× fragmentation: the small tensors occupy separate segments that can't be merged to satisfy a large allocation — even though total free memory is sufficient.

```
Allocate 8×16 MiB → free all → allocate 4×32 MiB
  Expected: 128 MiB reserved
  Actual:   256 MiB reserved (16 MiB blocks in separate segments can't merge)
```

**Why surprising:** "CUDA out of memory" often means "allocator can't find a contiguous segment," not "GPU actually full." The fix is allocation order (large first, small last) or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**T4 benchmark:** Allocate tensors small-to-large vs large-to-small. Compare `torch.cuda.memory_allocated()` vs `torch.cuda.memory_reserved()`. Try the same pattern with `expandable_segments:True` vs `False`.

**Expected magnitude:** Up to 2× memory waste from bad order.

### 5.2 `empty_cache()` doesn't release cuBLAS workspace

**The surprise:** `torch.cuda.empty_cache()` frees PyTorch's caching allocator pool but NOT cuBLAS workspace allocations. These accumulate silently, especially with repeated CUDA graph captures. `torch._C._cuda_clearCublasWorkspaces()` exists but is undocumented.

**Why surprising:** The "empty cache" function doesn't empty everything. GPU memory leaks that `nvidia-smi` shows but `torch.cuda.memory_allocated()` doesn't.

**T4 benchmark:** Repeatedly capture a CUDA graph containing matmul. Monitor `torch.cuda.memory_allocated()` and `nvidia-smi` before/after `empty_cache()`.

**Expected magnitude:** Modest for single captures; accumulates with repeated captures.

---

## 6. Mixed Precision & Tensor Cores

### 6.1 FP16 `eps=1e-8` rounds to zero — silent NaN generator

**The surprise:** FP16 can't represent values smaller than ~6×10⁻⁵. Adam's default `eps=1e-8` with FP16 mixed precision rounds epsilon to zero, producing NaN gradients. The fix is `eps=1e-4` or higher.

**Why surprising:** A hyperparameter that doesn't matter in FP32 silently breaks training in FP16. No error, no warning — just NaN after ~50-200 steps.

**T4 benchmark:** Train a small model with AMP, compare `eps=1e-8` vs `eps=1e-4`. Observe when NaN appears.

**Expected magnitude:** Training diverges within 50-200 steps with FP32-default epsilon.

### 6.2 Tensor Core utilization on T4: 15-30% of theoretical peak

**The surprise:** T4 Tensor Cores offer 65 TFLOPS FP16 theoretical vs 8.1 TFLOPS FP32 — an 8× cliff. But real models achieve only 3-4× speedup because the bottleneck is memory bandwidth (not compute), and Tensor Cores only help compute-bound layers. Small matmul dims (<2048) see dramatically lower utilization.

Real T4 benchmarks:

| Model | FP32 | Mixed Precision | Actual Speedup |
|---|---|---|---|
| BERT-Large | 16 sent/s | 63 sent/s | 3.93× |
| ResNeXt101 | 161 img/s | 598 img/s | 3.71× |
| BERT-Base | 51 sent/s | 193 sent/s | 3.75× |

**Why surprising:** 65 TFLOPS vs 8.1 TFLOPS = 8× theoretical → 3.75× actual. The missing 4.25× is memory bandwidth bottleneck + launch overhead + non-tensor-core ops.

**T4 benchmark:** Compare FP32 vs AMP for a range of matmul sizes (256×256 to 8192×8192). Show where Tensor Cores kick in and how utilization changes with size.

**Expected magnitude:** 1.5-4× for AMP-enabled workloads (vs FP32).

---

## 7. Data Loading

### 7.1 DataLoader with `num_workers` is 50-124× slower than direct indexing for in-memory data

**The surprise:** For datasets fully in RAM, DataLoader with `num_workers>0` triggers massive IPC overhead (~200,000 context switches in 40 seconds). Accessing the same data via direct tensor indexing is up to 124× faster. The workers compete with each other and the main process for CPU cores.

**Why surprising:** The standard recommendation ("always use DataLoader with num_workers=4") is catastrophically wrong for in-memory data.

**T4 benchmark:** Compare DataLoader (workers=0,2,4,8) vs manual tensor slicing for an in-memory dataset (e.g., CIFAR-10 loaded into RAM). Measure throughput and GPU idle time.

**Expected magnitude:** 50-124× slowdown with workers on in-memory data.

### 7.2 DataLoader workers: diminishing returns past 4-8

**The surprise:** More workers can be slower. Setting `num_workers` to `cpu_count()` (common advice) guarantees oversubscription — the main process also needs a core. The optimum is typically `max(0, cpu_count - num_gpus - 2)`, not `cpu_count`.

On Colab T4 VMs (typically 2-4 vCPUs), `num_workers=2` is usually optimal; 4+ degrades.

**T4 benchmark:** Sweep `num_workers=0,1,2,4,8` for a disk-backed dataset. Find the knee where throughput plateaus or drops.

**Expected magnitude:** Degradation past 4-8 workers on typical Colab CPUs.

---

## 8. `torch.compile` & JIT

### 8.1 `torch.compile` can be slower than eager on T4

**The surprise:** On T4 with small models, `torch.compile` can be 21% *slower* and use 62% *more memory* than eager execution (verified in `projects/systems/torch-compile-pipeline/`). The compilation overhead (seconds to minutes) may never amortize on short training runs.

Inductor on T4 (SM 7.5, Turing) is at a disadvantage: Triton's CUDA codegen targets SM 8.0+ (Ampere/Hopper), and cuDNN's hand-tuned heuristics beat Triton's generic kernels on T4.

**Why surprising:** The standard advice ("use torch.compile for a free speedup") is wrong on T4. Eager mode is often faster.

**T4 benchmark:** Compare eager vs `torch.compile` for CNN, MLP, and Transformer of varying sizes. Include compile time in the total. Already partially done in `projects/systems/torch-compile-pipeline/`.

**Expected magnitude:** 0.8-2× (slower to faster depending on model size and architecture).

### 8.2 Non-contiguous `max()` under `torch.compile` is 8× slower

**The surprise:** `torch.max(x)` when `x` is non-contiguous triggers a 3-stage reduction kernel instead of 2-stage under `torch.compile`. Same operation, same values — 8× slower purely due to tensor layout.

**Why surprising:** `torch.compile` amplifies layout sensitivity. A `.contiguous()` call before `.max()` can be the difference between inductor winning (2× faster) and losing (8× slower).

**T4 benchmark:** Compare `torch.max(x)` vs `torch.max(x.contiguous())` under `torch.compile` for transposed tensors of varying sizes.

**Expected magnitude:** 2-8× slowdown for `max()` on non-contiguous under compile.

---

## 9. Deterministic & Correctness Modes

### 9.1 Deterministic mode makes `F.interpolate` extremely slow

**The surprise:** `torch.use_deterministic_algorithms(True)` + `F.interpolate(mode='bilinear')` triggers a slow reference kernel instead of the fast cuDNN implementation. The deterministic path can be 10-100× slower.

**Why surprising:** "Deterministic = slightly slower" is the expectation. "Deterministic = 100× slower" is reality for some ops.

**T4 benchmark:** Time `F.interpolate` with and without deterministic mode for bilinear and nearest modes.

**Expected magnitude:** 10-100× slowdown for bilinear interpolate in deterministic mode.

### 9.2 `model.eval()` alone does nothing for memory

**The surprise:** `model.eval()` disables dropout/batchnorm special behavior but does NOT disable autograd. PyTorch still builds the computation graph and stores intermediate activations. Without `torch.no_grad()`, inference on a model that fits during training can OOM.

**Why surprising:** Half the "inference optimization" (`.eval()`) does nothing for memory. Users OOM during inference and blame the model size.

**T4 benchmark:** Run inference on a moderate model with `.eval()` only vs `.eval()` + `torch.no_grad()`. Compare peak memory.

**Expected magnitude:** 20-50% higher memory without `torch.no_grad()`.

---

## 10. Indexing & Scatter Operations

### 10.1 `scatter_add` performance depends on index distribution — 10-100× gap

**The surprise:** `index_put_` with `accumulate=True` uses either a sort-based algorithm or atomic adds depending on the index distribution. With many duplicate indices (common in embedding gradients), the atomic path is 10-100× faster, but PyTorch's dispatch heuristic doesn't always pick it.

**T4 benchmark:** Compare `scatter_add` with uniform indices (few duplicates) vs concentrated indices (many duplicates) on 1M-element tensors.

**Expected magnitude:** 10-100× slower for the wrong dispatch path.

---

## Benchmark Pipeline — Ready to Implement

The following are the top candidates from this catalog that can be benchmarked with simple scripts on T4, ranked by surprise-to-effort ratio:

| # | Experiment | Code complexity | Expected magnitude |
|---|---|---|---|
| 1 | `.to(device, dtype)` combined vs two-step | 3 lines | 3-8× |
| 2 | CPU vs GPU crossover (matmul size sweep) | 20 lines | 2-100× |
| 3 | Hidden `.item()` sync tax | 30 lines | 10-30% |
| 4 | Implicit `.contiguous()` copies (chain ops) | 15 lines | 2-10× |
| 5 | `cross_entropy` vs `log_softmax+gather` | 10 lines | 10-30× |
| 6 | CUDA first-call tax | 10 lines | 50,000× first vs nth |
| 7 | `torch.where` vs boolean masking | 10 lines | 2-5× |
| 8 | Allocator fragmentation (order + expandable_segments) | 20 lines | 2× |
| 9 | Deterministic interpolate cost | 5 lines | 10-100× |
| 10 | FP16 eps=1e-8 NaN generator | 30 lines | Divergence |
| 11 | `index_select` vs regular indexing | 10 lines | 2-6× |
| 12 | `torch.compile` + non-contiguous max | 10 lines | 2-8× |

---

## What We Already Benchmarked

| Topic | Status | Reference |
|---|---|---|
| GPU→CPU transfer asymmetry | Done | `docs/reference/pytorch-cpu-gpu-transfer.md` |
| `torch.compile` pipeline on T4 | Done | `projects/systems/torch-compile-pipeline/` |
| Pinned memory DMA mechanism | Done | `projects/systems/pytorch-transfer-benchmark/` |

## Sources

- PyTorch GitHub issues: #108968, #120996, #141822, #63618, #166114, #50879, #108521, #144431, #184084, #35901, #183806, #187094, #100850, #181238, #172537, #185819, #185886, #173313
- NVIDIA CUDA Programming Guide: Page-Locked Host Memory, CUDA Graphs, Tensor Cores
- Andrej Karpathy: GPU benchmarking methodology
- Colab T4 verified benchmarks from this repository
