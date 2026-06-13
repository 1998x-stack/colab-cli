# torch.compile Pipeline Analysis

Benchmark the three `torch.compile` backends to isolate what each compilation stage contributes: Dynamo graph capture, AOTAutograd traced backward, and Triton kernel compilation.

## Pipeline stages

Each backend adds one more stage to the compilation stack:

| Backend | Dynamo graph capture | Autograd (backward) | Kernel backend | What you're testing |
|---------|---------------------|---------------------|----------------|---------------------|
| `eager` (no compile) | -- | eager autograd | cuDNN/cuBLAS | Baseline — raw PyTorch |
| `eager` | FX graph capture | eager autograd | cuDNN/cuBLAS | Dynamo overhead alone |
| `aot_eager` | FX graph capture | AOTAutograd traced | cuDNN/cuBLAS | +traced backward graph |
| `inductor` | FX graph capture | AOTAutograd traced | Triton fused kernels | +kernel fusion & codegen |

## Results — BenchmarkCNN (1.2M params) on Tesla T4

2026-06-13, Colab free-tier T4, `torch==2.6.0+cu128`, 64×3×64×64 input, 100 measured iterations per config.

| Backend | Compile (s) | Fwd (samp/s) | Fwd+Bwd (samp/s) | Peak Mem (MB) | vs Baseline |
|---------|------------|-------------|-------------------|---------------|-------------|
| eager (no compile) | 0.0 | 7,808 | 2,303 | 273 | 1.00x |
| eager | 2.3 | 7,770 | 2,389 | 272 | **1.04x** |
| aot_eager | 2.7 | 7,275 | 2,346 | 273 | **1.02x** |
| inductor | 19.1 | 5,606 | 1,815 | 442 | **0.79x** |

## Analysis

### Inductor regressed — and that's the point

On this model+GPU combination, full `torch.compile(backend="inductor")` was **21% slower** and used **62% more memory** than raw eager execution. Three mechanisms:

1. **Model too small (1.2M params).** Kernel launch overhead dominates Triton's compute savings. Fusion eliminates intermediate reads/writes but each fused kernel takes longer to dispatch — on a model this shallow, the dispatch cost exceeds the memory-bandwidth savings.

2. **T4 is SM 7.5 (Turing-era).** Triton's CUDA codegen targets SM 8.0+ (Ampere/Hopper). On T4, cuDNN's hand-tuned heuristics (`cudnnFindConvolutionForwardAlgorithm`) pick layouts optimized for this specific hardware. Triton generates generic kernels that don't exploit T4's tensor-core quirk of fp32 accumulation.

3. **Triton workspace allocation.** `torch.compile` allocates persistent workspace buffers for kernel intermediates. Eager mode frees intermediates after each op. For a 272 MB eager workload, inductor holds 442 MB — the extra 170 MB is Triton's scratch space.

### AOTAutograd's contribution is real but subtle

Comparing `eager` (1.04x fwd+bwd) vs `aot_eager` (1.02x) — the traced backward in `aot_eager` is marginally *slower* here because this model's autograd graph is linear (residual connections are simple additions). AOTAutograd pays off when the backward graph has complex branching (e.g., multi-head attention, nested control flow) where the eager autograd engine spends time traversing the tape.

### Dynamo's FX graph capture costs ~0.5% throughput

`eager` backend at 7,770 vs 7,808 fwd samples/s — a 0.5% drop from graph capture overhead. This is noise-level for a model this size. On larger models (100M+ params), the graph capture overhead amortizes to zero while the fusion opportunities grow.

## When does inductor win?

This benchmark establishes the **failure regime**. Inductor wins in the **success regime**:

| Factor | This benchmark (fail) | Success regime |
|--------|----------------------|----------------|
| Model size | 1.2M params | 10M+ params |
| Arithmetic intensity | Low (3×3 convs) | High (matmul-heavy, transformers) |
| Batch size | 64 | 128+ |
| GPU | T4 (SM 7.5) | A100 (SM 8.0) / H100 (SM 9.0) |
| Forward/backward ratio | ~3.4:1 (conv-dominated) | ~2:1 or lower (matmul benefits more from fused backward) |

The rule of thumb: inductor helps when the GPU spends more time *computing* than *launching*. Small conv nets on T4 are launch-bound.

## Reproducing

```bash
# Provision + upload + run
colab new --gpu T4 -s compile-bench
colab upload train.py /content/train.py
colab exec -f launch.py --timeout 120
colab exec -f check_progress.py --timeout 15

# Download results (tar on VM first — colab download doesn't do directories)
echo 'import subprocess; subprocess.run(["tar","-czf","/content/out.tar.gz","-C","/content","torch-compile-pipeline-output"])' | colab exec -s compile-bench --timeout 15
colab download -s compile-bench /content/out.tar.gz ./output/out.tar.gz
tar -xzf output/out.tar.gz -C output/
```

### Testing on larger models

Change the model definition in `train.py` to stress the compiler more. A simple transformer encoder block will shift the results toward inductor's favor:

```python
# Replace BenchmarkCNN with this to see inductor win:
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
model = nn.TransformerEncoder(encoder_layer, num_layers=4)
# Input: (B, seq_len=64, d_model=512)
```

Expected: inductor ~1.3-1.8x throughput on transformer workloads, with the gain coming almost entirely from fused multi-head attention (flash attention via Triton).

## Artifacts

- `metrics.csv` — raw numbers per backend
- `compile_pipeline_comparison.png` — 4-panel bar chart (throughput, memory, compile time)
- `pipeline_stages.png` — annotated pipeline diagram showing which stages each backend runs
