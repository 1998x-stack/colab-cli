# Prefill vs Decode: Two-Phase Inference

Transformer inference with KV cache splits into two distinct phases with very different compute profiles.

## Prefill Phase

**What happens:** The full prompt is processed in a single forward pass. All prompt tokens'
K and V are computed and cached. The model generates logits for the entire sequence.

**Compute characteristics:**
- **Compute-bound:** Processes many tokens at once -> high GPU utilization
- **Matrix-matrix operations:** Q, K, V projections over the full prompt length
- **Causal mask applies:** Each token can only attend to positions <= its own
- **No KV cache yet:** This is the step that populates the cache

**Duration:** Proportional to prompt length. For a 2048-token prompt on T4: ~200-500ms.

## Decode Phase

**What happens:** One new token is generated per step. Its K and V are projected and appended
to the cache. Attention is computed against ALL cached keys and values.

**Compute characteristics:**
- **Memory-bound:** Only 1 token -> GPU compute units are underutilized
- **Matrix-vector operations:** Q projection is 1xD, attention is 1x(cached_length)
- **No masking needed:** The single query attends to all cached positions
- **Cache I/O dominates:** Reading the full KV cache from HBM is the bottleneck

**Duration:** Roughly constant per step (dominated by cache reads). On T4: ~5-20ms/step.

## The Bottleneck Shift

```
Prefill:  GPU compute >>> memory bandwidth
Decode:   memory bandwidth >>> GPU compute
```

This is why large-batch inference is efficient for prefill but not decode:
- Prefill benefits from batching (more tokens -> higher GPU utilization)
- Decode with batch=B must read B separate KV caches -> linear memory growth

## Continuous Batching

Modern serving systems (vLLM, TGI) use continuous batching to overlap prefill and decode:

1. New requests enter prefill together -> high GPU utilization
2. Requests that finish prefill transition to decode -> each gets its own cache
3. Multiple decode-phase requests share a batch -> amortizes the cost

This is more efficient than static batching where all requests must finish together.

## Latency vs Throughput

| | Latency-sensitive | Throughput-focused |
|---|---|---|
| **Prefill** | Process quickly (user waiting) | Batch many prompts |
| **Decode** | Generate token-by-token (streaming) | Overlap multiple decodes |

**Tradeoff:** Longer prompts -> more KV cache memory -> fewer concurrent requests.
Shorter prompts -> less cache -> higher throughput.

## Practical Numbers (T4, 4-layer GPT, d_model=256)

| Phase | Time | % of step |
|-------|------|-----------|
| Prefill (256 tokens) | ~50ms | 100% |
| Decode (step 1) | ~8ms | 16% of prefill |
| Decode (step 50) | ~10ms | 20% of prefill |
| Decode (step 200) | ~15ms | 30% of prefill |

Decode time increases slowly with cache size (linear growth), but remains much faster
than full recomputation at each step.
