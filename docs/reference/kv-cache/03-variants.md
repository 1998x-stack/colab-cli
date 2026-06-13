# KV Cache Variants: GQA, MQA, MLA, PagedAttention

The standard KV cache stores one K,V pair per attention head. As models scale,
this becomes a memory bottleneck. Several variants reduce cache memory.

## Multi-Query Attention (MQA)

**Idea:** Share a single K and V across ALL attention heads. Only Q has multiple heads.

**Memory:** $2 \times L \times S \times d_k$ instead of $2 \times L \times H \times S \times d_k$ -> Hx reduction.

**Used in:** PaLM, older models.

**Tradeoff:** Significant memory savings, but quality degrades. K/V represent only one
"view" of the input. Used when extreme memory efficiency is needed.

## Grouped-Query Attention (GQA)

**Idea:** Split heads into G groups. Each group shares one K,V. G=1 is MQA; G=H is MHA.

$$\text{K heads} = G, \quad \text{V heads} = G, \quad \text{Q heads} = H$$

**Memory:** $2 \times L \times G \times S \times d_k$ -- Gx less than MHA.

**Used in:** Llama 2 70B (G=8), Llama 3, Mistral.

**Why it works well:** Most attention heads capture redundant patterns. A few K/V views
are sufficient. GQA recovers most of MHA's quality with 2-8x less KV memory.

**Configuration examples:**
| Model | H (Q heads) | G (KV groups) | KV cache savings |
|-------|------------|---------------|-----------------|
| Llama 2 7B | 32 | 32 (MHA) | 1x |
| Llama 2 70B | 64 | 8 | 8x |
| Mistral 7B | 32 | 8 | 4x |
| Llama 3 70B | 64 | 8 | 8x |

## Multi-head Latent Attention (MLA)

**Idea (DeepSeek-V2):** Compress K and V into a low-rank latent space, then decompress
during attention. Far more aggressive than GQA.

$$K_{latent} = W_{down}^K \cdot X \in \mathbb{R}^{d_{latent}}$$
$$K = W_{up}^K \cdot K_{latent} \in \mathbb{R}^{H \times d_k}$$

**Memory:** Stores only the latent vector per token -> $L \times S \times d_{latent}$.
With $d_{latent} = 512$ and $H \times d_k = 128 \times 128 = 16384$, this is a 32x reduction.

**Used in:** DeepSeek-V2, DeepSeek-V3.

**Challenge:** The up-projection adds compute during decode. DeepSeek absorbs this into
the Q-projection via a mathematical equivalence, making it compute-neutral.

## PagedAttention (vLLM)

**Idea:** KV cache is NOT a contiguous tensor per request. Instead, it's stored in
fixed-size "pages" (blocks), like virtual memory.

**Key concepts:**
- **Block size:** Fixed-size chunks (e.g., 16 tokens per block)
- **Block table:** Maps logical positions -> physical blocks
- **Non-contiguous:** Blocks can be scattered across GPU memory

**Benefits:**
1. **Zero fragmentation:** No wasted memory between requests (internal fragmentation only in the last block)
2. **Sharing:** Prefix caches (system prompts, few-shot examples) can share physical blocks
3. **Dynamic allocation:** Blocks are allocated on-demand, not pre-allocated for max length

**Memory utilization:** PagedAttention achieves ~96% KV cache utilization vs ~20-40% for
contiguous allocation (due to over-provisioning for max_length).

## Comparison Table

| Variant | Memory per layer | Quality impact | Used in |
|---------|-----------------|----------------|---------|
| MHA (baseline) | $2HSd_k$ | Best | GPT-2, BERT |
| GQA (G groups) | $2GSd_k$ | Minor (<1%) | Llama 2/3, Mistral |
| MQA (G=1) | $2Sd_k$ | Noticeable (~1-2%) | PaLM |
| MLA | $2Sd_{latent}$ | Minor (with careful design) | DeepSeek-V2/V3 |
| PagedAttention | N/A (allocation, not format) | None | vLLM |
