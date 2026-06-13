# KV Cache: Mechanism & Math

## The Problem

In autoregressive decoding, each new token attends to ALL previous tokens. Without caching,
at step t we recompute K and V for tokens 0..t-1, then compute attention. This repeats work:
step t recomputes everything step t-1 already computed, plus one new token.

**Cost without cache:**
- Step 1: compute K,V for 1 token -> O(1)
- Step 2: compute K,V for 2 tokens -> O(2)
- ...
- Step N: compute K,V for N tokens -> O(N)
- **Total: O(N^2)** time, O(N^2) memory for attention matrix

## The Solution: KV Cache

Keys and values from past tokens don't change. Store them and reuse.

**Cost with cache:**
- Prefill (step 1): compute K,V for all prompt tokens -> O(L_prompt)
- Each decode step: compute K,V for 1 new token, append to cache -> O(1)
- **Total: O(N)** time per step after prefill

## Mathematical Derivation

### Standard Attention (no cache)

For query Q, key K, value V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

At step t of autoregressive generation, we have tokens $x_0, x_1, ..., x_t$. To predict $x_{t+1}$:

$$K = [K_0, K_1, ..., K_t] \in \mathbb{R}^{(t+1) \times d_k}$$
$$V = [V_0, V_1, ..., V_t] \in \mathbb{R}^{(t+1) \times d_v}$$
$$Q = Q_t \in \mathbb{R}^{1 \times d_k}$$

The attention computes:

$$\text{Attention}(Q_t, K, V) = \text{softmax}\left(\frac{Q_t [K_0, ..., K_t]^T}{\sqrt{d_k}}\right) [V_0, ..., V_t]$$

$K_0...K_{t-1}$ were already computed at step t-1. We recompute them unnecessarily.

### With KV Cache

Store the accumulated K and V:

1. **Step t (prefill):** Project all prompt tokens to get $K_{0..t}, V_{0..t}$. Store in cache.
2. **Step t+1 (decode):** Project only the new token: $k_{t+1}, v_{t+1} \in \mathbb{R}^{1 \times d_k}$.
   Append to cache:
   $$K_{cache} \leftarrow [K_{cache}; k_{t+1}]$$
   $$V_{cache} \leftarrow [V_{cache}; v_{t+1}]$$
   Compute attention with $Q_{t+1}$ against $K_{cache}, V_{cache}$.

## Memory Analysis

For a model with L layers, H heads, sequence length S, head dimension $d_k$:

| | Without Cache | With Cache |
|---|---|---|
| **Per-step KV compute** | $O(S \cdot H \cdot d_k)$ | $O(H \cdot d_k)$ |
| **Peak memory (KV)** | $O(S^2)$ (attention matrix) | $O(L \cdot H \cdot S \cdot d_k)$ |
| **Total time (N tokens)** | $O(N^2)$ | $O(N)$ per step |

### Concrete Example

GPT-2 small (12 layers, 12 heads, $d_k$=64, bf16):

| Sequence length | Cache memory | Attention matrix |
|---|---|---|
| 256 | 12 x 12 x 256 x 64 x 2 = 4.7 MB | 256^2 x 12 x 2 = 1.6 MB |
| 1024 | 18.9 MB | 25 MB |
| 4096 | 75.5 MB | 403 MB |
| 16384 | 302 MB | 6.4 GB |

Memory grows linearly with S for KV cache, but quadratically for the attention matrix.

## Limitations

- **Memory still grows with S.** At very long sequences (100K+), KV cache dominates GPU memory.
  Variants like GQA (fewer K/V heads) and PagedAttention help.
- **Batch x cache.** When batching multiple requests, each has its own cache.
  Continuous batching systems manage this carefully.
- **Precision.** KV cache is typically stored in fp16/bf16. Int8/int4 quantization
  can reduce memory ~2-4x with minimal accuracy loss (see KIVI, KVQuant).
