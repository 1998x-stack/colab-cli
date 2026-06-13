# vLLM Model Comparison Tutorial

Benchmarks 3 small LLMs with vLLM on a Colab T4 GPU (16GB VRAM). Compares latency (TTFT), throughput (tokens/sec), and peak VRAM across models with different quantization strategies.

## Models

| Model | Quantization | Approx VRAM |
|-------|-------------|-------------|
| Qwen2.5-7B-Instruct | AWQ 4-bit | ~4.5 GB |
| Llama-3.2-3B-Instruct | FP16 | ~6.5 GB |
| SmolLM2-1.7B-Instruct | FP16 | ~3.5 GB |
| Phi-3.5-mini-instruct (fallback) | FP16 | ~7.5 GB |

## Quick Start

```bash
# Provision T4 VM
colab new --gpu T4 -s vllm-compare

# Upload files
colab upload launch.py launch.py
colab upload compare.py compare.py

# Run (gated Llama model needs HF_TOKEN)
colab exec -f launch.py --timeout 120

# Check progress
colab upload check_progress.py check_progress.py
colab exec -f check_progress.py --timeout 15

# Download results
colab download /content/results.json .

# Clean up
colab stop -s vllm-compare
```

## Requirements

- Colab session with T4 GPU (free tier works)
- HuggingFace token for gated models (Llama-3.2-3B). Set via `HF_TOKEN` env var or the Colab secrets UI.
- If Llama is inaccessible, Phi-3.5-mini is used as fallback.

## Expected Results (approximate, T4)

| Model | TTFT (ms) | Tok/s | Peak VRAM (GB) |
|-------|-----------|-------|----------------|
| Qwen2.5-7B-AWQ | ~45 | ~85 | ~4.8 |
| Llama-3.2-3B | ~30 | ~120 | ~6.5 |
| SmolLM2-1.7B | ~15 | ~200 | ~3.5 |

## Tutorial Takeaways

- AWQ quantization (Qwen 7B) fits a 7B model on a T4 at 4.5GB VRAM — competitive with much smaller FP16 models.
- vLLM's PagedAttention enables efficient memory management even on constrained GPUs.
- Smaller models offer lower latency but weaker reasoning — choose based on your use case.
