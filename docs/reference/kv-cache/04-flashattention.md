# KV Cache + FlashAttention: Compatibility & Interaction

FlashAttention is an I/O-aware exact attention algorithm. It fuses the attention computation
into a single CUDA kernel, avoiding materializing the $S \times S$ attention matrix in HBM.

## How They Interact

KV cache and FlashAttention solve different problems:

- **KV cache:** Avoids recomputing K and V for past tokens
- **FlashAttention:** Makes computing attention from K,V faster and more memory-efficient

They are **complementary**, not alternatives. Modern inference uses both:
1. KV cache stores past K,V (avoids recomputation)
2. FlashAttention computes the attention efficiently (reduces memory bandwidth)

## Training: FlashAttention Without Cache

During training, we process full sequences (no cache). FlashAttention shines here:
- Standard attention materializes $[B, H, S, S]$ attention matrix in HBM -> $O(S^2)$ memory
- FlashAttention computes attention in tiles in SRAM -> $O(S)$ memory
- Training speedup: 3-7x for long sequences (4096+)

## Inference: Both Together

**With KV cache + FlashAttention:**
1. Cache stores $[B, H, S_{cache}, d_k]$ K and V
2. New token's K,V ($[B, H, 1, d_k]$) are appended
3. FlashAttention computes attention over the full cached sequence

**Compatibility:** FlashAttention-2 and FlashAttention-3 both support KV cache inference.
The `flash_attn_with_kvcache` function in FlashAttention's API directly handles this:

```python
from flash_attn import flash_attn_with_kvcache

# q: [B, 1, H, d_k] -- new token
# k_cache, v_cache: [B, H, S_cache, d_k] -- accumulated cache
# k_new, v_new: [B, 1, H, d_k] -- new token's K,V

output = flash_attn_with_kvcache(
    q, k_cache, v_cache, k_new, v_new,
    causal=False  # no mask needed for single query token
)
# k_cache, v_cache are updated in-place
```

## Interleaving: When to Recompute vs Cache

For very long sequences where KV cache exceeds GPU memory, there's a **recomputation tradeoff**:

| Strategy | Memory | Compute |
|----------|--------|---------|
| Cache everything | $O(LS)$ | $O(S)$ per step |
| Cache nothing (recompute) | $O(1)$ | $O(S^2)$ per step |
| **Sliding window cache** | $O(LW)$ | $O(W)$ per step |
| **StreamingLLM** (keep attention sinks) | $O(L(W+4))$ | $O(W+4)$ per step |

**Sliding window** (Mistral, Mixtral): Only cache the last W tokens. Older tokens are dropped.
This gives O(W) memory and compute regardless of sequence length.

**StreamingLLM:** Keep the first ~4 tokens (attention sinks) + sliding window.
Attention sinks absorb attention weight and prevent perplexity collapse when
dropping middle tokens.

## FlashAttention Versions

| Version | KV Cache Support | Key Feature |
|---------|-----------------|-------------|
| v1 | Via separate API | I/O-aware tiling |
| v2 | `flash_attn_with_kvcache` | 2x faster, supports GQA natively |
| v3 | `flash_attn_with_kvcache` | Hopper FP8, async, 1.5-2x faster |

**PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`): Since PyTorch 2.0,
also supports FlashAttention backend and KV cache via separate K,V inputs. Does NOT
support in-place cache update -- pass the full concatenated K,V each step.
