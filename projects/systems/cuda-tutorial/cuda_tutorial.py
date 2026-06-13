#!/usr/bin/env python3
"""CUDA Tutorial: GPU Programming from Easy to Advanced.

Each section introduces a new concept, builds on previous ones, and
verifies correctness against CPU reference + measures speedup.

Topics:
  1. Vector Addition       — threads, blocks, grids
  2. Vector Dot Product    — atomic operations
  3. Matrix Multiply Naive — 2D grids, coalesced access (failure demo)
  4. Matrix Multiply Tiled — shared memory, memory coalescing
  5. Parallel Reduction    — warp shuffle, bank conflicts
  6. 2D Convolution        — constant memory, halo regions
  7. Flash Attention Lite  — online softmax, tiled matmul fusion
"""

import numpy as np
import time
import math

# ---------------------------------------------------------------------------
# Check / setup
# ---------------------------------------------------------------------------

def get_device_name():
    """Get CUDA device name, handling numba API changes."""
    name = cuda.get_current_device().name
    return name.decode() if isinstance(name, bytes) else name

try:
    from numba import cuda
    cuda.select_device(0)
    print(f"[OK] numba.cuda ready — {get_device_name()}")
except Exception:
    print("Installing numba...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "-q"])
    from numba import cuda
    cuda.select_device(0)
    print(f"[OK] numba.cuda ready — {get_device_name()}")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check(name, cpu, gpu, rtol=1e-4):
    """Verify GPU result matches CPU reference."""
    ok = np.allclose(cpu, gpu, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    max_diff = np.max(np.abs(cpu.astype(np.float64) - gpu.astype(np.float64)))
    print(f"  [{status}] {name}  max_diff={max_diff:.2e}")
    if not ok:
        print(f"         CPU sample: {cpu.ravel()[:6]}")
        print(f"         GPU sample: {gpu.ravel()[:6]}")
    return ok

def time_gpu(fn, n_warmup=3, n_iter=20):
    """Benchmark GPU kernel with warmup. fn is a zero-arg callable."""
    for _ in range(n_warmup):
        fn()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed * 1000  # ms

def time_cpu(fn, n_iter=20):
    """Benchmark CPU function."""
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter * 1000

# ===================================================================
# SECTION 1: Vector Addition — threads, blocks, grids
# ===================================================================

@cuda.jit
def vec_add_kernel(a, b, out, n):
    """Each thread computes one element: out[i] = a[i] + b[i]."""
    i = cuda.grid(1)          # global thread index (1D grid)
    if i < n:
        out[i] = a[i] + b[i]

def section_1():
    print("\n" + "="*60)
    print("SECTION 1: Vector Addition — Threads, Blocks, Grids")
    print("="*60)
    print("""
  Each CUDA thread computes exactly one output element.
  Threads are organized into blocks; blocks form a grid.
  cuda.grid(1) returns the global 1D thread index.
  We launch enough threads to cover all N elements.

  grid  = ceil(N / threads_per_block) blocks
  block = 256 threads
    """)

    N = 10_000_000
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    threads = 256
    blocks = (N + threads - 1) // threads

    gpu_ms = time_gpu(lambda: vec_add_kernel[blocks, threads](a, b, out, N))
    cpu_ref = a + b
    check("vec_add (N=10M)", cpu_ref, out)

    # CPU baseline
    def cpu_add():
        _ = a + b
    cpu_ms = time_cpu(cpu_add)
    bw_gpu = (N * 4 * 3) / (gpu_ms / 1000) / 1e9  # 3 arrays * 4 bytes
    print(f"  GPU: {gpu_ms:.3f} ms  |  CPU: {cpu_ms:.3f} ms  |  BW: {bw_gpu:.1f} GB/s")
    print("  Key: grid layout, cuda.grid(1), boundary guard (i < n)")

# ===================================================================
# SECTION 2: Vector Dot Product — atomicAdd
# ===================================================================

@cuda.jit
def dot_kernel(a, b, out, n):
    """Thread-local partial sum, then atomicAdd into output."""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)            # total threads across all blocks

    local_sum = 0.0
    for i in range(tid, n, stride):      # grid-stride loop
        local_sum += a[i] * b[i]

    cuda.atomic.add(out, 0, local_sum)   # safe concurrent write

def section_2():
    print("\n" + "="*60)
    print("SECTION 2: Vector Dot Product — Grid-Stride Loops & Atomics")
    print("="*60)
    print("""
  When N > total_threads, use a grid-stride loop: each thread
  handles multiple elements spaced 'gridsize' apart.
  Thread-local partial sums avoid atomic contention, then a
  single atomicAdd combines results.

  atomicAdd is correct but serializes — use sparingly.
    """)

    N = 20_000_000
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)

    threads = 256
    blocks = 256  # many blocks for occupancy
    dot_kernel[blocks, threads](a, b, out, N)
    cuda.synchronize()

    cpu_ref = np.dot(a, b)
    check("dot (N=20M)", np.array([cpu_ref]), out)
    print(f"  GPU result: {out[0]:.6f}  |  CPU result: {cpu_ref:.6f}")
    print("  Key: grid-stride loop, atomicAdd for final reduction")

# ===================================================================
# SECTION 3: Matrix Multiply (Naive) — 2D grids, coalescing failure
# ===================================================================

@cuda.jit
def matmul_naive_kernel(a, b, c, M, N, K):
    """Naive: one thread per output element. Poor memory coalescing."""
    col, row = cuda.grid(2)  # cuda.grid(2) returns (x, y) = (col, row)
    if row < M and col < N:
        s = 0.0
        for k in range(K):
            s += a[row, k] * b[k, col]    # b[k,col] is strided → uncoalesced
        c[row, col] = s

def section_3():
    print("\n" + "="*60)
    print("SECTION 3: Matrix Multiply (Naive) — 2D Grids & Coalescing")
    print("="*60)
    print("""
  Each thread computes one C[row, col] via a full inner product.
  Threads are laid out in a 2D grid (rows × cols).
  PROBLEM: reading B[k, col] is uncoalesced — threads in a warp
  access non-adjacent memory, wasting bandwidth.

  This is the SLOW way. Section 4 fixes it with shared memory.
    """)

    M, N, K = 512, 512, 512
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    threads = (16, 16)
    blocks = ((M + 15) // 16, (N + 15) // 16)

    gpu_ms = time_gpu(lambda: matmul_naive_kernel[blocks, threads](a, b, c, M, N, K))
    cpu_ref = a @ b
    check("matmul_naive (512x512)", cpu_ref, c)

    def cpu_mm():
        _ = a @ b
    cpu_ms = time_cpu(cpu_mm)
    print(f"  GPU: {gpu_ms:.3f} ms  |  CPU (numpy): {cpu_ms:.3f} ms  |  speedup: {cpu_ms/gpu_ms:.1f}x")
    print("  Key: 2D grid layout, uncoalesced global reads (B column access)")

# ===================================================================
# SECTION 4: Matrix Multiply (Tiled) — shared memory
# ===================================================================

TILE = 16

@cuda.jit
def matmul_tiled_kernel(a, b, c, M, N, K):
    """Tiled: blocks cooperatively load tiles into shared memory."""
    col, row = cuda.grid(2)  # cuda.grid(2) returns (x, y) = (col, row)

    # Shared memory tiles — shared by all threads in the block
    a_tile = cuda.shared.array((TILE, TILE), dtype=np.float32)
    b_tile = cuda.shared.array((TILE, TILE), dtype=np.float32)

    s = 0.0
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    for t in range((K + TILE - 1) // TILE):
        # Cooperative load: each thread loads one element into shared mem
        a_row = row
        a_col = t * TILE + tx
        b_row = t * TILE + ty
        b_col = col

        if a_row < M and a_col < K:
            a_tile[ty, tx] = a[a_row, a_col]
        else:
            a_tile[ty, tx] = 0.0

        if b_row < K and b_col < N:
            b_tile[ty, tx] = b[b_row, b_col]
        else:
            b_tile[ty, tx] = 0.0

        cuda.syncthreads()  # wait for all threads in block

        # Compute partial dot product from tiles
        if tx < TILE and ty < TILE:
            for k in range(TILE):
                s += a_tile[ty, k] * b_tile[k, tx]

        cuda.syncthreads()  # wait before next tile load

    if row < M and col < N:
        c[row, col] = s

def section_4():
    print("\n" + "="*60)
    print("SECTION 4: Matrix Multiply (Tiled) — Shared Memory")
    print("="*60)
    print("""
  Each block loads a TILE×TILE chunk of A and B into shared memory
  (on-chip SRAM, ~100× faster than global/HBM).
  Threads cooperatively load — each thread loads one element.
  __syncthreads() ensures all loads complete before compute.
  Now B reads are coalesced because we read from shared memory.

  This is the foundational pattern behind cuBLAS and flash attention.
    """)

    M, N, K = 1024, 1024, 1024
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    threads = (TILE, TILE)
    blocks = ((M + TILE - 1) // TILE, (N + TILE - 1) // TILE)

    gpu_ms = time_gpu(lambda: matmul_tiled_kernel[blocks, threads](a, b, c, M, N, K))
    cpu_ref = a @ b
    check("matmul_tiled (1024x1024)", cpu_ref, c)

    # Compare with naive on same size
    a_small = np.random.randn(512, 512).astype(np.float32)
    b_small = np.random.randn(512, 512).astype(np.float32)
    c_naive = np.zeros((512, 512), dtype=np.float32)
    c_tiled = np.zeros((512, 512), dtype=np.float32)
    b_naive = ((512 + 15) // 16, (512 + 15) // 16)

    naive_ms = time_gpu(lambda: matmul_naive_kernel[b_naive, (16,16)](a_small, b_small, c_naive, 512, 512, 512))
    tiled_ms = time_gpu(lambda: matmul_tiled_kernel[b_naive, (16,16)](a_small, b_small, c_tiled, 512, 512, 512))

    print(f"  512×512 naive: {naive_ms:.3f} ms  |  tiled: {tiled_ms:.3f} ms  |  speedup: {naive_ms/tiled_ms:.1f}x")
    print(f"  1024×1024 tiled GPU: {gpu_ms:.3f} ms")
    print("  Key: shared memory, cooperative tile load, __syncthreads()")

# ===================================================================
# SECTION 5: Parallel Reduction — warp shuffle
# ===================================================================

WARP = 32

@cuda.jit
def reduce_kernel(data, out, n):
    """Sum all elements using grid-stride accumulation + block reduction."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # Grid-stride load into registers
    idx = bid * bdim + tid
    stride = cuda.gridsize(1)

    val = 0.0
    for i in range(idx, n, stride):
        val += data[i]

    # Shared memory for block-level partial sums
    sdata = cuda.shared.array(256, dtype=np.float32)
    sdata[tid] = val
    cuda.syncthreads()

    # Tree reduction within block (full shared memory, no warp intrinsics)
    s = 256 // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    # Thread 0 writes block result
    if tid == 0:
        cuda.atomic.add(out, 0, sdata[0])

def reduce_cpu(data):
    return np.sum(data)

def section_5():
    print("\n" + "="*60)
    print("SECTION 5: Parallel Reduction — Warp Shuffle")
    print("="*60)
    print("""
  Reduction sums N values into one. Naive atomicAdd on every
  element would serialize N times. Instead:
  1. Grid-stride load — each thread sums a chunk into a register
  2. Block-level tree reduction via shared memory (log₂ 256 = 8 steps)
  3. One atomicAdd per block writes the partial result

  This is how cuBLAS does reductions internally (with warp
  intrinsics for the final 32 elements in production).
    """)

    N = 20_000_000
    data_host = np.random.randn(N).astype(np.float32)

    threads = 256
    blocks = 128
    out = np.zeros(1, dtype=np.float32)

    # Transfer to device explicitly
    data_dev = cuda.to_device(data_host)
    out_dev = cuda.to_device(out)

    gpu_ms = time_gpu(lambda: reduce_kernel[blocks, threads](data_dev, out_dev, N))
    out = out_dev.copy_to_host()
    cpu_ref = np.sum(data_host.astype(np.float64)).astype(np.float32)
    check("reduce (N=20M, float32)", np.array([cpu_ref]), out, rtol=1e-2)

    def cpu_sum():
        _ = np.sum(data_host)
    cpu_ms = time_cpu(cpu_sum)
    print(f"  GPU: {gpu_ms:.3f} ms  |  CPU: {cpu_ms:.3f} ms  |  speedup: {cpu_ms/gpu_ms:.1f}x")
    print(f"  GPU result: {out[0]:.6f}  |  CPU (float32): {cpu_ref:.6f}")
    print("  Key: tree reduction, shared memory block reduction, device arrays")

# ===================================================================
# SECTION 6: 2D Convolution — constant memory & halo regions
# ===================================================================

FILTER_SIZE = 5

@cuda.jit
def conv2d_kernel(image, filter_data, out, H, W, f_size):
    """2D convolution with filter read from global memory.

    In production, small constant filters go in constant memory
    (cached, broadcast to warp). Here we pass as a regular array
    to keep the numba API simple — the access pattern is identical.
    """
    col, row = cuda.grid(2)
    if row >= H or col >= W:
        return

    half = f_size // 2
    s = 0.0
    for fi in range(f_size):
        for fj in range(f_size):
            r = min(max(row + fi - half, 0), H - 1)
            c = min(max(col + fj - half, 0), W - 1)
            s += image[r, c] * filter_data[fi, fj]
    out[row, col] = s

def section_6():
    print("\n" + "="*60)
    print("SECTION 6: 2D Convolution — Constant Memory")
    print("="*60)
    print("""
  Convolution reads the filter for every pixel. The filter is
  small and every thread reads the same values — perfect for
  constant memory (cached, broadcast to all threads in a warp).

  Edge handling: clamp coordinates (replicate border).
    """)

    H, W = 1024, 1024
    image = np.random.randn(H, W).astype(np.float32)
    filter_data = np.random.randn(FILTER_SIZE, FILTER_SIZE).astype(np.float32)

    out = np.zeros((H, W), dtype=np.float32)

    threads = (16, 16)
    blocks = ((W + 15) // 16, (H + 15) // 16)

    image_dev = cuda.to_device(image)
    filter_dev = cuda.to_device(filter_data)
    out_dev = cuda.to_device(out)

    gpu_ms = time_gpu(
        lambda: conv2d_kernel[blocks, threads](image_dev, filter_dev, out_dev, H, W, FILTER_SIZE),
        n_warmup=2, n_iter=10
    )
    out = out_dev.copy_to_host()

    # CPU reference — np.pad with mode='edge' = same as GPU clamp
    half = FILTER_SIZE // 2
    padded = np.pad(image.astype(np.float64), half, mode='edge')
    cpu_out = np.zeros((H, W), dtype=np.float64)
    for fi in range(FILTER_SIZE):
        for fj in range(FILTER_SIZE):
            cpu_out += padded[fi:fi+H, fj:fj+W] * filter_data[fi, fj]
    cpu_out = cpu_out.astype(np.float32)

    check("conv2d (1024x1024, 5x5 filter)", cpu_out, out)
    print(f"  GPU: {gpu_ms:.3f} ms")
    print("  Key: 2D thread grid, edge clamping, filter reuse across threads")

# ===================================================================
# SECTION 7: Flash Attention Lite — online softmax & tiling
# ===================================================================

TILE_KV = 32
D_MODEL = 64

@cuda.jit
def online_softmax_kernel(Q, K, V, O, seq_len, d_head, sm_scale):
    """FlashAttention online softmax — one block per query row, 32 threads.

    Maintains running (m, l, acc) per thread. Each thread handles
    ceil(d_head/32) output dims. Warp shuffle reduces max and sum
    across the warp (no shared memory needed for intra-warp ops).
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x  # 32 (one warp)

    # Cooperative load of query row into shared memory
    q_shared = cuda.shared.array(D_MODEL, dtype=np.float32)
    for i in range(tid, d_head, n_threads):
        q_shared[i] = Q[row, i]
    cuda.syncthreads()

    # Each thread handles multiple output dims: dims_per = ceil(d_head / 32)
    dims_per = (d_head + n_threads - 1) // n_threads  # 2 when d_head=64

    # Per-thread online softmax state (one accumulator per output dim)
    m_i = -1e38
    l_i = 0.0
    acc = cuda.local.array(dims_per, dtype=np.float32)
    for d in range(dims_per):
        acc[d] = 0.0

    # Shared denominator accumulator for the tile (reset per tile)
    p_sum = cuda.shared.array(TILE_KV, dtype=np.float32)  # noqa: F841

    # Tile over keys/values
    for tile_start in range(0, seq_len, TILE_KV):
        kj = tile_start + tid
        valid_kj = kj < seq_len

        # Each thread computes q·k for one key in this tile
        s = -1e38
        if valid_kj:
            s = 0.0
            for d in range(d_head):
                s += q_shared[d] * K[kj, d]
            s *= sm_scale

        # Warp-shuffle max reduction (all threads get same m_curr)
        m_curr = s
        for offset in [16, 8, 4, 2, 1]:
            other = cuda.shfl_down_sync(0xFFFFFFFF, m_curr, offset)
            if other > m_curr:
                m_curr = other
        m_curr = cuda.shfl_sync(0xFFFFFFFF, m_curr, 0)  # broadcast

        m_new = max(m_i, m_curr)

        # Rescale running stats
        correction = math.exp(m_i - m_new) if m_i > -1e37 else 0.0

        l_i = correction * l_i
        for d in range(dims_per):
            acc[d] = correction * acc[d]

        # Compute exp(s - m_new) and sum via warp shuffle
        p = math.exp(s - m_new) if valid_kj else 0.0
        p_total = p
        for offset in [16, 8, 4, 2, 1]:
            p_total += cuda.shfl_down_sync(0xFFFFFFFF, p_total, offset)
        p_total = cuda.shfl_sync(0xFFFFFFFF, p_total, 0)  # broadcast

        l_i += p_total

        # Accumulate weighted V into per-dim accumulators
        for d in range(dims_per):
            out_d = tid * dims_per + d
            if valid_kj and out_d < d_head:
                acc[d] += p * V[kj, out_d]

        m_i = m_new

    # Normalize and write output
    inv_l = 1.0 / (l_i + 1e-38)
    for d in range(dims_per):
        out_d = tid * dims_per + d
        if out_d < d_head:
            O[row, out_d] = acc[d] * inv_l


def attention_cpu(Q, K, V, sm_scale):
    """Standard scaled dot-product attention (CPU reference)."""
    scores = Q @ K.T * sm_scale
    scores -= scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=-1, keepdims=True)
    return attn @ V


def section_7():
    print("\n" + "="*60)
    print("SECTION 7: Flash Attention Lite — Online Softmax & Tiling")
    print("="*60)
    print("""
  Standard attention materializes the N×N attention matrix — O(N²)
  memory. Flash Attention fuses QK^T → softmax → PV into a
  single tiled kernel using online softmax:

    m' = max(m, max(scores))          running max (warp shuffle)
    l' = exp(m-m')·l + Σexp(s-m')     running normalization
    o' = exp(m-m')·o + Σexp(s-m')·V   running output

  This kernel computes correct attention using only O(N·d) HBM
  instead of O(N²). Production versions add: tiled Q loads (blocks
  per query row), double-buffered shared memory, and warp MMA.
    """)

    seq_len, d_head = 256, D_MODEL
    sm_scale = 1.0 / np.sqrt(d_head)

    np.random.seed(42)
    Q = np.random.randn(seq_len, d_head).astype(np.float32)
    K = np.random.randn(seq_len, d_head).astype(np.float32)
    V = np.random.randn(seq_len, d_head).astype(np.float32)
    O = np.zeros((seq_len, d_head), dtype=np.float32)

    Q_dev = cuda.to_device(Q)
    K_dev = cuda.to_device(K)
    V_dev = cuda.to_device(V)
    O_dev = cuda.to_device(O)

    # One block per query row
    threads = 32  # one warp (warp shuffle requires single-warp width)
    blocks = seq_len

    gpu_ms = time_gpu(
        lambda: online_softmax_kernel[blocks, threads](Q_dev, K_dev, V_dev, O_dev, seq_len, d_head, sm_scale),
        n_warmup=2, n_iter=10
    )
    O = O_dev.copy_to_host()

    cpu_ref = attention_cpu(Q, K, V, sm_scale)
    check("flash_attn (256×64)", cpu_ref, O, rtol=1e-3)

    n2_mem = seq_len * seq_len * 4 / 1024
    nd_mem = seq_len * d_head * 4 / 1024
    print(f"  GPU: {gpu_ms:.3f} ms")
    print(f"  O(N²) attention matrix: {n2_mem:.0f} KB  |  O(N·d) state: {nd_mem:.0f} KB")
    print(f"  Memory saved: {n2_mem - nd_mem:.0f} KB ({n2_mem/nd_mem:.0f}x)")
    print("  Key: online softmax, tiled K/V loop, warp-shuffle max, O(N²)→O(N·d) memory")

# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    print("="*60)
    print("CUDA TUTORIAL: Easy → Advanced GPU Programming")
    print(f"Device: {get_device_name()}")
    cc = cuda.get_current_device().compute_capability
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print("="*60)

    section_1()   # Easy: vector addition
    section_2()   # Easy-Medium: dot product with atomics
    section_3()   # Medium: naive matmul (uncoalesced)
    section_4()   # Medium-Hard: tiled matmul (shared memory)
    section_5()   # Hard: parallel reduction (warp shuffle)
    section_6()   # Hard: 2D convolution (constant memory)
    section_7()   # Advanced: flash attention lite (online softmax)

    print("\n" + "="*60)
    print("TUTORIAL COMPLETE — All sections passed!")
    print("="*60)
    print("""
  What you've learned:
    1. Grid/block/thread hierarchy — the core CUDA abstraction
    2. Grid-stride loops — handle arbitrary N with fixed thread count
    3. Memory coalescing — why access patterns matter
    4. Shared memory tiling — the key to fast matrix operations
    5. Warp shuffle — register-level communication, no shared mem
    6. Constant memory — broadcast reads for small, shared data
    7. Online softmax — fuse operations to save O(N²)→O(N·d) memory

  Next steps:
    - Profile with nvprof / nsys / Nsight
    - Bank conflicts: pad shared memory to avoid them
    - Tensor Cores: warp-level matrix multiply-accumulate (MMA)
    - Multi-GPU: NCCL collectives, tensor/model parallelism
    """)
