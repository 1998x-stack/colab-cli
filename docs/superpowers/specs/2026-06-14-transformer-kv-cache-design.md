# Transformer with KV Cache — Design Spec

**Date:** 2026-06-14
**Status:** approved

## Overview

Deep-dive KV cache technical report (5 docs) + pluggable transformer project demonstrating KV cache speedup on character-level language modeling (tiny Shakespeare). Deploy to Colab with 2-minute cron artifact fetching.

## Project structure

```
docs/reference/kv-cache/
├── 01-mechanism.md          # Math: attention → KV cache derivation
├── 02-prefill-vs-decode.md  # Two-phase inference: latency vs throughput
├── 03-variants.md           # GQA, MQA, MLA, PagedAttention
├── 04-flashattention.md     # FlashAttention compatibility
└── 05-benchmarks.md         # Measured speedup + memory analysis

projects/nlp/transformer-kv-cache/
├── config.py                # All hyperparams + model config
├── attention.py             # MHA + KV-cache-aware causal MHA
├── kv_cache.py              # KVCache data structure
├── model.py                 # Transformer assembly
├── train.py                 # Training loop + logging + metrics
├── generate.py              # Inference: with/without KV cache comparison
├── charts.py                # Training curves + inference speedup charts
├── launch.py                # Colab bootstrap
├── check_progress.py        # Monitor script
└── fetch.sh                 # Cron pull script (2-min interval)
```

## Architecture

### Model

Decoder-only transformer (GPT-style), char-level tokenizer.

Config-driven (config.py): n_layer=4, n_head=4, d_model=256, d_ff=1024, block_size=256, dropout=0.1, vocab_size=65. All tunable via CLI.

### Attention (attention.py)

Training: standard causal MHA, no cache.
Inference: `CausalMHAWithCache.forward(x, kv_cache=None)` — accepts optional KVCache. Projects only new token's KQV, appends to cache.

### KVCache (kv_cache.py)

```python
@dataclass
class KVCache:
    k: Tensor | None  # [B, H, L, D]
    v: Tensor | None  # [B, H, L, D]
    def update(self, k_new, v_new) -> tuple[Tensor, Tensor]
    def reset(self)
    @property
    def seq_len(self) -> int
```

### Model assembly (model.py)

Builds blocks from attention + KVCache. Swappable attention implementations via config.

### Training (train.py)

Dataset: tiny Shakespeare (char-level). Outputs:
- `logs/train.log` — per-epoch, timestamped, self-contained lines
- `pngs/training_curves.png` — loss + perplexity + time/epoch
- `pngs/kv_cache_speedup.png` — inference latency vs seq_len (with/without cache)
- `metrics.csv` — epoch, loss, perplexity, tokens_per_sec, elapsed_s

### Generation (generate.py)

Compares two modes:
- Without cache: recompute full attention each step → O(L^2) per token
- With cache: compute only new token's KQV, append to cache → O(L) per token

Plots latency vs sequence length for both modes.

## Colab deployment

### Flow

1. launch.py: package → upload to Colab → install deps → nohup train.py
2. Output dir: `/content/transformer-kv-cache-output/`
3. Weights-only checkpoint (<200MB) for download

### Cron fetch.sh (2-minute interval)

Each tick:
1. Check session alive: `colab sessions | grep <name>`
2. Tar output on VM via `colab exec`
3. Download tar via REST (`colab download`)
4. Extract → tail log last 5 lines + tail CSV last 3 lines
5. Report: done? current loss? ETA?

### Constraints

- Colab T4 free-tier GPU: ~10 min window
- Default config fits within this window (~8 min training)
- Proxy setup per CLAUDE.md proxy section
- Ruff lint must pass before deploy

## KV Cache docs (5 files)

All in `docs/reference/kv-cache/`:

1. **01-mechanism.md**: Attention formula → KV cache math derivation, O(L^2)→O(L), memory layout, prefill vs decode token cost
2. **02-prefill-vs-decode.md**: Two-phase inference deep dive, latency/throughput tradeoff, batching implications, continuous batching
3. **03-variants.md**: Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-head Latent Attention (MLA, DeepSeek-V2), PagedAttention (vLLM)
4. **04-flashattention.md**: How KV cache interacts with FlashAttention, tiling, recomputation strategies, interleaving
5. **05-benchmarks.md**: Measured speedup data from this project, memory analysis, latency vs seq_len plots
