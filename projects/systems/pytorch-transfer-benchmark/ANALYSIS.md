# PyTorch CPU-GPU Transfer Performance: Root Cause Analysis & Benchmark

**Tested on:** Tesla T4 (Colab free tier), PyTorch 2.11.0+cu128, CUDA 12.8

## TL;DR

| Transfer Direction | Default `.to()` | Pinned / `non_blocking=True` | Speedup |
|---|---|---|---|
| CPU → GPU | 6.9 GB/s | 12.4 GB/s | 1.8× |
| GPU → CPU | **1.4 GB/s** | **13.3 GB/s** | **9.2×** |

GPU→CPU with default `.to('cpu')` is **9× slower** than it should be. Two one-line fixes exist.

---

## 1. The Asymmetry Problem

PyTorch's `.to()` looks symmetric but isn't:

```python
x_gpu = x_cpu.to("cuda")   # fast: ~6.9 GB/s
x_cpu = x_gpu.to("cpu")    # SLOW: ~1.4 GB/s  ← 5-9× slower!
```

The same operation in reverse is drastically slower. On our T4 benchmark, a 2 GB tensor takes:

| Method | Time | Throughput |
|---|---|---|
| `.to('cuda')` | 313 ms | 6.9 GB/s |
| `.to('cpu')` | **1,492 ms** | **1.4 GB/s** |
| `.to('cpu', non_blocking=True)` | 161 ms | 13.3 GB/s |

**1.5 seconds** to copy 2 GB from GPU to CPU — that's slower than downloading it from the internet on a 10 Gbps link.

## 2. Root Cause: Non-Pinned Memory

### What happens under `.to('cpu')`

```
GPU VRAM ──DMA──>  [???]  ──CPU memcpy──>  Your tensor (pageable)
                       ↑
                  Problem: DMA can't target pageable memory directly
```

CUDA's DMA engine can only write to **pinned (page-locked) memory** — pages that the OS guarantees won't be swapped out. When you call `.to('cpu')` on a non-pinned destination, the CUDA driver must:

1. **Allocate** a temporary pinned staging buffer
2. **DMA** GPU → staging buffer (fast)
3. **memcpy** staging buffer → your pageable tensor (slow, CPU-bound)
4. **Free** the staging buffer

Steps 1, 3, and 4 are pure overhead. Step 3 is a CPU-driven copy that runs at memory bandwidth (~20-30 GB/s) rather than PCIe DMA bandwidth. The allocation in step 1 also serializes with other CUDA operations.

### Why it doesn't affect CPU→GPU

CPU→GPU uses the **opposite direction**: the source is pageable memory, the destination is VRAM. The GPU can read from pageable memory via DMA (with IOMMU/BAR translation), or the driver does a similar staging buffer trick internally. On modern systems with IOMMU, the GPU DMA reads directly from pageable memory — but even with a staging buffer, the fast path (GPU reads from staging buffer via DMA) is less bottlenecked than the slow path (CPU copies to pageable memory).

### The T4 PCIe constraint

T4 uses PCIe 3.0 x16: theoretical max **15.75 GB/s** per direction. Our measured 13.3 GB/s pinned throughput achieves **84% of theoretical** — reasonable after protocol overhead (TLP headers, ACK packets, etc.).

## 3. The Fix: Two Approaches

### Fix A: `non_blocking=True` (simplest)

```python
# Before: 1.4 GB/s
x_cpu = x_gpu.to("cpu")

# After: 13.1 GB/s
x_cpu = x_gpu.to("cpu", non_blocking=True)
torch.cuda.synchronize()  # must sync before reading x_cpu
```

**Why it works:** `non_blocking=True` forces PyTorch to use an internal pinned staging buffer because the operation is asynchronous — the tensor can't be read until the copy completes, so PyTorch must use DMA. The CUDA driver allocates a pinned buffer, does the DMA transfer, and maps the result to your tensor.

**Tradeoff:** Each call allocates+free a pinned buffer internally. For one-off transfers this is fine. For tight loops, use Fix B.

### Fix B: Explicit pinned memory (zero-allocation for repeated transfers)

```python
# Pre-allocate once
x_cpu_pinned = torch.empty(size, dtype=dtype, pin_memory=True)

# Each transfer: 13.1 GB/s
x_cpu_pinned.copy_(x_gpu)
torch.cuda.synchronize()
```

**Why it works:** `pin_memory=True` allocates the tensor in page-locked memory via `cudaHostAlloc()`. `copy_()` from a GPU tensor detects the pinned destination and uses direct DMA — no staging buffer, no CPU memcpy.

**Tradeoff:** Pinned memory is a scarce resource. Too many large pinned tensors degrade system performance because the OS can't swap those pages. For repeated transfers (e.g., data loading), pre-allocating a pinned buffer pool is ideal.

### Both fixes compared

| Approach | Throughput | Allocations per transfer | Best for |
|---|---|---|---|
| `.to('cpu')` | 1.4 GB/s | 1 temp | Never |
| `.to('cpu', nb=True)` | 13.1 GB/s | 1 temp pinned | One-shot transfers |
| Pinned `copy_()` | 13.3 GB/s | 0 (pre-alloc) | Repeated transfers, DataLoader |

## 4. Full Benchmark Results (T4, 2 GB float16 tensor)

### CPU → GPU

| Method | Throughput (GB/s) | Latency (ms) |
|---|---|---|
| `.to('cuda')` | 6.9 | 313 |
| `.to('cuda', non_blocking=True)` | 6.8 | 316 |
| Pinned `.to('cuda')` | **12.4** | 174 |
| Pinned `.to('cuda', non_blocking=True)` | 12.4 | 174 |

**CPU→GPU pinned also helps** — 1.8× improvement. This is T4-specific; consumer GPUs with PCIe 4.0/5.0 see less benefit because the DMA-from-pageable path is faster.

### GPU → CPU

| Method | Throughput (GB/s) | Latency (ms) |
|---|---|---|
| `.to('cpu')` | **1.4** | 1,492 |
| `.to('cpu', non_blocking=True)` | 13.3 | 161 |
| Pinned `.copy_(gpu)` | 13.3 | 161 |
| Pinned `.copy_(gpu, non_blocking=True)` | **13.4** | 160 |

### Throughput vs tensor size (GPU→CPU)

| Size | Default `.to('cpu')` | `non_blocking=True` | Pinned `copy_()` |
|---|---|---|---|
| 1 MB | 1.5 GB/s | 9.3 GB/s | 8.4 GB/s |
| 10 MB | 2.7 GB/s | 12.3 GB/s | 12.0 GB/s |
| 100 MB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 500 MB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 1 GB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 2 GB | 1.4 GB/s | 13.3 GB/s | 13.3 GB/s |

Small tensors (<100 MB) have lower efficiency due to fixed kernel launch + CUDA API overhead (~0.1 ms per call). Above 100 MB, throughput stabilizes.

## 5. Why This Matters

### Real-world impact

- **Training loop checkpointing:** Saving a 2 GB model state_dict every N steps. With default `.to('cpu')` that's 1.5s of GPU idle time per checkpoint. With pinned: 0.16s.
- **Data loading:** Each batch transferred GPU→CPU for logging/metrics. 100 batches/sec × 10 MB = 1 GB/s — default path already saturated, pinned gives headroom.
- **Inference serving:** Moving outputs from GPU→CPU for response serialization. At 1.4 GB/s, a 200 MB KV-cache takes 143 ms — nearly the entire latency budget for real-time serving.
- **Multi-GPU training:** gradient sync across GPUs often involves CPU staging. Default path adds seconds per synchronization step.

### The PyTorch design choice

PyTorch could default to `non_blocking=True` for `.to('cpu')` but doesn't because:
1. `non_blocking=True` requires the user to synchronize before reading — silent data races otherwise
2. It changes memory allocation behavior (pinned memory is limited)
3. Backward compatibility — code written for the blocking default could break

## 6. Recommendations

### For one-off transfers
```python
x_cpu = x_gpu.to("cpu", non_blocking=True)
torch.cuda.synchronize()
```

### For DataLoader / repeated transfers
```python
# Allocate pinned buffer once
pinned_buf = torch.empty(batch_shape, dtype=dtype, pin_memory=True)

# In training loop
pinned_buf.copy_(gpu_tensor)
# ... use pinned_buf on CPU
```

### For checkpoint saving
```python
# Pre-allocate pinned buffer for state dict copying
state_cpu = {k: torch.empty_like(v, device='cpu', pin_memory=True) 
             for k, v in model.state_dict().items()}
for k, v in model.state_dict().items():
    state_cpu[k].copy_(v)
torch.cuda.synchronize()
torch.save(state_cpu, "checkpoint.pt")
```

## 7. Reproducing

```bash
# Local (needs GPU)
python benchmark_transfer.py

# Colab (T4)
colab new --gpu T4 -s bench
colab upload benchmark_transfer.py /content/
colab exec -f launch.py --timeout 120
colab download -s bench /content/transfer-bench-results.tar.gz ./
```

Full benchmark script and raw data at `projects/systems/pytorch-transfer-benchmark/`.

## References

- [PyTorch Docs: Memory Pinning](https://pytorch.org/docs/stable/data.html#memory-pinning)
- [NVIDIA: How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [CUDA Programming Guide: Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
- Original article that motivated this benchmark: [Zhihu - PyTorch GPU↔CPU 传输缺陷](https://zhuanlan.zhihu.com/p/264178514)
