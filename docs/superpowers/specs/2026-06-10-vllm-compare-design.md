# vLLM Model Comparison Tutorial — Design Spec

## Goal

A Colab tutorial that benchmarks 3 small LLMs with vLLM on a T4 GPU, comparing latency, throughput, and VRAM usage. Educational focus: teach vLLM concepts (PagedAttention, continuous batching, AWQ quantization) through hands-on measurement.

## Files

| File | Purpose |
|------|---------|
| `compare.py` | Main script. Downloads models via huggingface_hub, runs vLLM benchmarks, prints comparison table, saves `/content/results.json`. |
| `launch.py` | Colab launcher. Installs vLLM + deps, spawns `compare.py` as detached subprocess. |
| `check_progress.py` | Tails log, shows current model being benchmarked, lists saved results. |
| `README.md` | Local reference: what this does, how to run, expected results. |

## Models

| Model | Quant | Approx VRAM | HuggingFace ID |
|-------|-------|-------------|----------------|
| Qwen2.5-7B-Instruct | AWQ 4-bit | ~4.5 GB | Qwen/Qwen2.5-7B-Instruct-AWQ |
| Llama-3.2-3B-Instruct | FP16 | ~6.5 GB | meta-llama/Llama-3.2-3B-Instruct |
| SmolLM2-1.7B-Instruct | FP16 | ~3.5 GB | HuggingFaceTB/SmolLM2-1.7B-Instruct |

All fit within T4's 16GB VRAM. Downloaded once, benchmarked sequentially, unloaded between runs.

## Metrics

- **TTFT** (time-to-first-token): latency for the first token to appear
- **TPS** (tokens per second): throughput on a batch of prompts
- **Peak VRAM**: max GPU memory during inference (reported by torch.cuda)
- **Load time**: seconds from model instantiation to ready

## Tutorial Flow (compare.py)

1. **Environment check** — GPU info, CUDA version, free VRAM
2. **Install verification** — confirm vLLM import, print version
3. **Per-model loop**:
   - Download from HuggingFace (snapshot_download)
   - Initialize vLLM LLM instance
   - Warm up with 2 prompts
   - Benchmark: batch of 10 diverse prompts, measure TTFT, TPS, VRAM
   - Unload (del model, torch.cuda.empty_cache)
4. **Comparison table** — print formatted results, save to JSON
5. **Analysis** — printed takeaways: when each model wins, quantization tradeoffs

## Prompt Set (same for all models)

1. "Explain what PagedAttention is in one paragraph." (technical explanation)
2. "Write a Python function to compute Fibonacci numbers recursively." (code generation)
3. "Summarize: 'The transformer architecture, introduced in 2017, revolutionized NLP by replacing recurrence with self-attention...'" (summarization)
4. "What is the capital of France and what is it known for?" (factual QA)
5. "If a train leaves at 60 mph and another at 40 mph, how long until they meet if 200 miles apart?" (reasoning)
6. "Write a haiku about machine learning." (creative)
7. "Translate to French: 'Hello, how are you today?'" (translation)
8. "What is the difference between supervised and unsupervised learning?" (QA)
9. "List three advantages of using GPU over CPU for deep learning." (structured QA)
10. "Explain the chain rule in calculus with a simple example." (explanation)

## Fallback Model

If Llama-3.2-3B is inaccessible (no HF_TOKEN or access not granted), substitute with `microsoft/Phi-3.5-mini-instruct` (~3.8B, no gating, fits T4).

## Dependencies

Installed in launch.py: `vllm`, `huggingface_hub`, `torch`. vLLM ships prebuilt CUDA 12.x wheels compatible with Colab's runtime.

## Constraints

- T4 GPU (16GB VRAM), free-tier Colab session (~2-4h)
- vLLM from pip (no source build)
- Models downloaded from HuggingFace Hub (no local cache pre-seeding)
- Llama-3.2-3B requires HuggingFace token for gated access — user must set HF_TOKEN

## Error Handling

- If a model OOMs: skip it, note in results, continue to next
- If HF_TOKEN is missing for Llama: skip that model, note in results, continue
- If GPU not detected: print diagnostic and exit early
- Benchmark failures on individual prompts: record error, continue batch

## Output

`/content/results.json`:
```json
{
  "hardware": {"gpu": "T4", "vram_gb": 16, "cuda": "12.x"},
  "models": {
    "Qwen2.5-7B-Instruct-AWQ": {
      "load_time_s": 12.3,
      "ttft_avg_ms": 45.2,
      "tps_avg": 89.1,
      "peak_vram_gb": 4.8,
      "status": "ok"
    },
    ...
  }
}
```
