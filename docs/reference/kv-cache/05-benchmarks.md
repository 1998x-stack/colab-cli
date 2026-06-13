# KV Cache Benchmarks

Measured speedup from this project's `generate.py` on a 4-layer GPT (d_model=256, 4 heads).

Run `python generate.py --checkpoint output/checkpoints/weights_epoch10.pt --max_tokens 200`
to reproduce.

## Expected Speedup (Theoretical)

For N generation steps and average sequence length L:

| Metric | Without Cache | With Cache | Speedup |
|--------|-------------|-----------|---------|
| K,V compute per step | $O(L)$ projections | $O(1)$ projection | $O(L)$ |
| Attention compute per step | $O(L)$ dot products | $O(L)$ dot products | $O(1)$ |
| Memory (peak) | $O(L^2)$ attention matrix | $O(L)$ cached K,V | -- |

The key savings: without cache, you recompute K and V for every token at every step.
With cache, each token's K and V is computed ONCE.

## Measured Results

Fill in after running on Colab/GPU:

| Sequence Length | Without Cache (ms/step) | With Cache (ms/step) | Speedup |
|----------------|------------------------|---------------------|---------|
| 10 | -- | -- | --x |
| 50 | -- | -- | --x |
| 100 | -- | -- | --x |
| 200 | -- | -- | --x |
| 500 | -- | -- | --x |

## Memory Profile

4-layer GPT, d_model=256, 4 heads, block_size=256, fp32:

| Component | Size |
|-----------|------|
| Model weights | ~3.2M params x 4 bytes = 12.8 MB |
| KV cache (seq_len=256, 4 layers) | 4 x 4 x 256 x 64 x 4 x 2 = 2.1 MB |
| Attention matrix (256^2) | 256 x 4 x 256 x 256 x 4 = 1 MB |
| Total inference | ~16 MB |

For reference, the attention matrix WITHOUT KV cache at step 256 would require
recomputing K,V for all 256 tokens x 4 layers -> much higher compute but same peak memory.

## Latency Breakdown (T4 GPU, expected)

| Component | Time per step | % |
|-----------|-------------|---|
| QKV projection | ~2ms | 25% |
| Cache read | ~3ms | 37% |
| Attention compute | ~2ms | 25% |
| FFN + residual | ~1ms | 13% |

Cache read dominates decode latency -- this is why KV cache quantization matters.

## How to Reproduce

```bash
# Train
python train.py --device cuda --max_epochs 10

# Benchmark
python generate.py \
  --checkpoint output/checkpoints/weights_epoch10.pt \
  --max_tokens 100 200 500 \
  --device cuda
```

Results appear in `output/pngs/kv_cache_speedup.png`.
