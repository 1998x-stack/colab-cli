# vLLM RAG Pipeline Tutorial

End-to-end RAG (Retrieval-Augmented Generation) pipeline on a Colab T4 GPU. Serves Qwen2.5-7B-AWQ via vLLM's OpenAI-compatible API, indexes SQuAD v2 passages in ChromaDB, and evaluates 100 questions with retrieval + generation metrics.

## Architecture

```
[vLLM API Server :8000] ←── [RAG Client]
                              ├─ Embed (all-MiniLM-L6-v2)
                              ├─ Retrieve (ChromaDB, top-3)
                              └─ Generate (vLLM completions)
```

## Quick Start

```bash
# Provision T4 VM
colab new --gpu T4 -s vllm-rag

# Upload files
colab upload launch.py launch.py
colab upload server.py server.py
colab upload rag_eval.py rag_eval.py

# Run (SQuAD v2 is public, no token needed)
colab exec -f launch.py --timeout 180

# Check progress
colab upload check_progress.py check_progress.py
colab exec -f check_progress.py --timeout 15

# Download results
colab download /content/rag_results.json .

# Clean up
colab stop -s vllm-rag
```

## Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Retrieval | Recall@1 | Fraction where top-1 passage contains answer |
| Retrieval | Recall@3 | Fraction where any top-3 passage contains answer |
| Retrieval | MRR | Mean Reciprocal Rank of first relevant passage |
| Generation | Exact Match | Fraction where answer contains ground truth |
| Generation | Token F1 | Token-level precision/recall vs ground truth |
| System | Avg latency | Wall time per query (embed+retrieve+generate) |
| System | Peak VRAM | Max GPU memory during run |

## Expected Results (approximate, T4)

| Metric | Value |
|--------|-------|
| Recall@1 | ~0.60 |
| Recall@3 | ~0.82 |
| MRR | ~0.71 |
| Exact Match | ~0.55 |
| Token F1 | ~0.68 |
| Avg latency | ~2.5s/query |
| Peak VRAM | ~5 GB |
