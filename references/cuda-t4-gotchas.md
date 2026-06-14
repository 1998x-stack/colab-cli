# CUDA T4 Gotchas

T4-specific limitations and CUDA programming traps verified on Colab T4 (CUDA 12.8, PyTorch 2.11.0+cu128).

## T4 Hardware Limits

| Spec | Value |
|------|-------|
| Compute capability | 7.5 (Turing) |
| VRAM | 15.6 GB |
| FP16 tensor cores | Yes (65 TFLOPS theoretical) |
| BF16 support | No (SM 7.5 < 8.0) |
| Flash Attention 3 | No |

### Tensor Core utilization peaks at 37%

T4's 65 TFLOPS FP16 theoretical peak is unreachable for single matmul. Peak observed: 23.8 TFLOPS at 3072×3072 (37%). FP16 is still 6.4× faster than FP32 at large sizes.

For max throughput: target matmul dims ≥768, use FP16, avoid mixed small/large matmuls.

### GPU is NOT always faster — know the crossover

- Matmul: GPU wins at ~128×128. At 64×64, CPU is faster.
- Element-wise ops (relu, add): GPU wins above ~100K elements. Below that, kernel launch overhead dominates.

Small-tensor workloads (token-level ops, small batch linears) may be faster on CPU.

## CUDA Dark Corners

### Cross-entropy + permute: 17-118× slower for LLM logits

`F.cross_entropy(logits.permute(0, 2, 1), targets)` triggers slow reduction kernels. At B=1, S=128, V=50257: CE+permute = 51.6 ms, log_softmax+gather = 0.45 ms (114.7× faster).

Fix:
```python
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
```

`.contiguous()` after permute does NOT help — bottleneck is the reduction kernel path.

### CUDA timing: perf_counter without synchronize() is 3-15× wrong

`time.perf_counter()` without `torch.cuda.synchronize()` measures CPU kernel launch latency (~15-70µs), not GPU execution time. Always use `torch.cuda.Event` for GPU benchmarks.

### CUDA 12.8 / PyTorch 2.11 fixed multiple "classic" traps

| Trap | Old expectation | Actual (CUDA 12.8) | Root cause |
|------|----------------|--------------------|------------|
| Implicit `.contiguous()` copies | 2-10× slowdown | 1.0-1.1× | Stride-aware kernels improved |
| `index_select` 2-6× slower | 2-6× | 1.0-1.2× | Gather kernel optimized |
| FP16 eps=1e-8 NaN | NaN in <200 steps | No NaN in 500 steps | GradScaler prevents underflow |
| CUDA first-call tax ~1.6s | 1.6s | ~389ms | Init path optimization |
| `torch.compile` 8× worse on non-contiguous max() | 8× | 1.0× | Inductor 3-stage→2-stage reduction |

Always verify known traps on your target CUDA/PyTorch version.

### view() crashes on permuted tensors; reshape() doesn't

After permute/transpose: `.view()` requires contiguous memory → `RuntimeError`. `.reshape()` copies internally when needed → always safe.

### DataLoader num_workers>0 hangs on Colab

With Rust-backed `tokenizers` library and `num_workers=2`, DataLoader stalls silently. `num_workers=0` fixes it. Pre-tokenization eliminates throughput concern.

### torch.compile on T4: no bf16, check compatibility

`torch.compile(model)` with bfloat16 on T4 emits warning and falls back to eager. Skip compilation on T4 or use float16.

## VRAM Management

### CUDA OOM during eval even when training fits

Beam search allocates extra tensors. Solutions:
- `torch.cuda.empty_cache()` before eval
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Smaller beam size during training eval

### First Colab session: CUDA JIT overhead

CUDA JIT compilation on first batch: 2-3 min. Model appears "stuck" but GPU is active at 77%+. Check `nvidia-smi`. Second session (cached JIT) starts instantly.

### Checkpoint size

- Full (model + optimizer + scheduler): ~1 GB for 61M param model
- Weights-only: ~120-233 MB
- With torch.compile artifacts: checkpoints bloat (42 MB → 126 MB for 10.75M model)
