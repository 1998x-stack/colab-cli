# PyTorch CPU-GPU Transfer: Why `.to('cpu')` is 9× Slower Than It Should Be

Date: 2026-06-14 | GPU: Tesla T4 | PyTorch 2.11.0+cu128 | CUDA 12.8

## The problem

```python
x_gpu = x_cpu.to("cuda")   # ~6.9 GB/s — fine
x_cpu = x_gpu.to("cpu")    # ~1.4 GB/s — 5-9× slower!
```

PyTorch's `.to()` appears symmetric but is drastically asymmetric in performance. GPU→CPU with default `.to('cpu')` transfers at **1.4 GB/s** on T4 — barely faster than spinning disk reads. The same tensor copied GPU→CPU with the right approach hits **13.3 GB/s** (PCIe 3.0 x16 line rate).

## Root cause: pageable memory can't receive DMA

CUDA's DMA engine can only write directly to **pinned (page-locked) memory** — pages the OS guarantees won't be swapped or relocated. Standard `torch.empty()` returns pageable memory, which the OS can move at any time.

When you call `.to('cpu')` on pageable memory, the CUDA driver inserts a hidden 4-step pipeline:

```
GPU VRAM ──DMA──> [pinned staging buffer] ──CPU memcpy──> Your tensor (pageable)
                      ↑ allocated + freed per call        ↑ CPU-driven, not DMA
```

1. **cudaHostAlloc** — allocate temporary pinned staging buffer
2. **cudaMemcpyAsync** (DMA) — GPU → staging buffer (fast, ~13 GB/s)
3. **memcpy** — staging buffer → your pageable tensor (CPU-bound, limited by system memory bandwidth)
4. **cudaFreeHost** — free staging buffer

Step 3 is the bottleneck. The CPU-driven `memcpy` runs at system memory bandwidth (~20-30 GB/s for a single core), but the real cost is the **synchronization**: the CPU must wait for the DMA to complete, then execute the memcpy, blocking the calling thread. The allocation in step 1 also serializes with the CUDA stream.

### Why CPU→GPU is faster without pinned memory

The reverse direction (CPU→GPU) uses the GPU as the DMA *target*. Modern systems with IOMMU (VT-d/AMD-Vi) allow the GPU to DMA-read directly from pageable memory via address translation. Even without IOMMU, the driver's internal staging buffer on the *source* side is less damaging because the GPU reads from it at full DMA speed — the bottleneck is the PCIe link, not the CPU.

On T4 (PCIe 3.0 x16), CPU→GPU default achieves 6.9 GB/s — about 44% of theoretical. The pinned path reaches 12.4 GB/s (79% of theoretical), showing that IOMMU/pageable DMA on T4 still incurs overhead.

### T4 PCIe constraint

T4 uses PCIe 3.0 x16: 15.75 GB/s theoretical max per direction. Our measured 13.3 GB/s achieves **84%** of line rate — the remaining 16% is PCIe protocol overhead (TLP headers, ACK/NACK, flow control).

Consumer GPUs (RTX 3090/4090) on PCIe 4.0/5.0 see proportionally higher absolute numbers but the same relative pattern.

## Benchmark data (Colab T4, 2 GB float16 tensor)

### GPU → CPU (the critical direction)

| Method | Throughput | Latency | Allocations |
|---|---|---|---|
| `.to('cpu')` | **1.4 GB/s** | 1,492 ms | 1 temp pinned |
| `.to('cpu', non_blocking=True)` | 13.3 GB/s | 161 ms | 1 temp pinned |
| Pinned `copy_(gpu)` | 13.3 GB/s | 161 ms | 0 (pre-alloc) |
| Pinned `copy_(gpu, non_blocking=True)` | 13.4 GB/s | 160 ms | 0 (pre-alloc) |

### CPU → GPU (less severe but still meaningful)

| Method | Throughput | Latency |
|---|---|---|
| `.to('cuda')` | 6.9 GB/s | 313 ms |
| Pinned `.to('cuda')` | 12.4 GB/s | 174 ms |

### Throughput vs tensor size (GPU→CPU only)

| Size | Default `.to('cpu')` | `non_blocking=True` | Pinned `copy_()` |
|---|---|---|---|
| 1 MB | 1.5 GB/s | 9.3 GB/s | 8.4 GB/s |
| 10 MB | 2.7 GB/s | 12.3 GB/s | 12.0 GB/s |
| 100 MB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 500 MB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 1 GB | 1.4 GB/s | 13.1 GB/s | 13.1 GB/s |
| 2 GB | 1.4 GB/s | 13.3 GB/s | 13.3 GB/s |

Small tensors (<100 MB) show lower efficiency due to fixed CUDA kernel launch overhead (~0.1 ms/call). Above 100 MB, throughput stabilizes.

## `.to('cpu', non_blocking=True)` — the surprising fast path

On CUDA 12.8 / PyTorch 2.11, `.to('cpu', non_blocking=True)` achieves the same ~13.3 GB/s as explicit pinned memory — **without the user allocating pinned memory**. Why:

When `non_blocking=True`, the operation is asynchronous — the returned tensor cannot be safely read until `torch.cuda.synchronize()` or a CUDA event fires. PyTorch exploits this: it allocates an internal pinned staging buffer, performs GPU→staging DMA, then maps the result to the returned tensor. The staging buffer is freed after the synchronize point.

This is *not* documented behavior and may change across PyTorch versions. The explicit pinned `copy_()` approach is the guaranteed-fast path.

## Fix A: `non_blocking=True` (one-shot transfers)

```python
# Before: 1.4 GB/s, blocks the CPU for 1.5s
x_cpu = x_gpu.to("cpu")

# After: 13.3 GB/s, 161 ms, non-blocking
x_cpu = x_gpu.to("cpu", non_blocking=True)
torch.cuda.synchronize()  # required before reading x_cpu
```

**Tradeoff:** Each call internally allocates+frees a pinned staging buffer. Acceptable for occasional transfers. Not for tight loops.

## Fix B: Explicit pinned memory (repeated transfers, zero allocation)

```python
# Allocate pinned buffer once
x_cpu_pinned = torch.empty(size, dtype=dtype, pin_memory=True)

# Each transfer: direct DMA, no internal allocation
x_cpu_pinned.copy_(x_gpu)
torch.cuda.synchronize()
```

**Why `copy_()` not `.to('cpu')` on a pinned tensor:** `.to('cpu')` on a pinned source tensor creates a new (pageable) output tensor — you lose the pinned destination. `copy_()` into a pre-allocated pinned tensor keeps the DMA path.

**Tradeoff:** Pinned memory locks physical pages. Too many large pinned tensors degrade system paging performance. Pre-allocate a fixed pool and reuse.

## For checkpointing specifically

```python
# Pre-allocate once, reuse every checkpoint
state_pinned = {
    k: torch.empty_like(v, device="cpu", pin_memory=True)
    for k, v in model.state_dict().items()
}

def save_checkpoint(model, path):
    for k, v in model.state_dict().items():
        state_pinned[k].copy_(v)
    torch.cuda.synchronize()
    torch.save(state_pinned, path)
```

2 GB checkpoint: 161 ms (pinned) vs 1,492 ms (default) — saves 1.3 seconds per save.

## When to use which

| Scenario | Method | Why |
|---|---|---|
| One-off `.to('cpu')` in inference | `non_blocking=True` | Simplest, no code restructuring |
| Checkpoint saving every N steps | Pre-allocated pinned `copy_()` | Zero per-save allocation |
| DataLoader worker output | `pin_memory=True` in DataLoader | Built-in, zero code change |
| Debug logging (small tensors) | Default `.to('cpu')` | Overhead negligible below 10 MB |
| Production training loop metrics | Pinned ring buffer | Reuse same buffers, no alloc |

## Why PyTorch doesn't fix the default

PyTorch could make `non_blocking=True` the default for `.to('cpu')`, but doesn't, because:

1. **Silent data races.** `non_blocking=True` returns a tensor that may not be ready. Users must synchronize before reading. Changing the default would break all existing code that reads the result immediately.
2. **Pinned memory is scarce.** The internal staging buffer consumes pinned pages. Making this the default would cause system-wide memory pressure in multi-process workloads (DataLoader workers).
3. **Backward compatibility.** A 10-year-old API with millions of lines of dependent code cannot change semantics silently.

The explicit opt-in (`non_blocking=True` or `pin_memory=True`) is the contract.

## Related

- Benchmark project + raw data: `projects/systems/pytorch-transfer-benchmark/`
- Original article: [Zhihu - PyTorch GPU↔CPU 传输缺陷](https://zhuanlan.zhihu.com/p/264178514)
- [PyTorch Docs: Memory Pinning](https://pytorch.org/docs/stable/data.html#memory-pinning)
- [CUDA Programming Guide: Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
- [NVIDIA Developer Blog: How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
