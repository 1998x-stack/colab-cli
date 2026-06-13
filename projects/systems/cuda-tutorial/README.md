# CUDA Tutorial: Easy to Advanced GPU Programming

Seven progressive CUDA kernel programming tutorials using numba.cuda, from vector addition through flash attention.

## Usage

```bash
# Local training (requires CUDA GPU + numba)
python cuda_tutorial.py

# Colab deployment
cb launch.py
```

## Sections

| # | Topic | Key Technique |
|---|-------|---------------|
| 1 | Vector Addition | Threads, blocks, grids (`cuda.grid(1)`) |
| 2 | Vector Dot Product | Grid-stride loops, `atomicAdd` |
| 3 | Matrix Multiply (Naive) | 2D grids, uncoalesced memory access |
| 4 | Matrix Multiply (Tiled) | Shared memory, cooperative tile load, `syncthreads` |
| 5 | Parallel Reduction | Tree reduction, warp shuffle (`shfl_down_sync`) |
| 6 | 2D Convolution | Constant memory, edge clamping |
| 7 | Flash Attention Lite | Online softmax, tiled KV loop, O(N^2) to O(N*d) memory |

## Key results

| Metric | Value |
|--------|-------|
| Vector add (10M) | BW measured in GB/s |
| Naive matmul 512x512 | Uncoalesced (slow) |
| Tiled matmul 512x512 | ~2-5x speedup over naive |
| Flash attention 256x64 | O(N*d) memory, verified vs CPU |

## Gotchas

- Requires a CUDA-capable GPU and `numba` (auto-installed if missing on Colab).
- The flash attention section uses a single warp (32 threads) for warp shuffle operations.
- Tiled matmul uses a fixed tile size of 16; performance varies by GPU compute capability.
- All sections verify correctness against CPU reference with `np.allclose`.
