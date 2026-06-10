# vLLM RAG Pipeline Tutorial — Design Spec

## Goal

A Colab tutorial that demonstrates a production RAG pipeline: serve a model with vLLM's OpenAI-compatible API, index SQuAD v2 passages, run 100 evaluation queries, and measure retrieval + generation quality. Educational focus: teach the embed → retrieve → generate pattern and how to evaluate RAG systems.

## Files

| File | Purpose |
|------|---------|
| `server.py` | Starts vLLM with Qwen2.5-7B-AWQ in OpenAI-compatible API server mode on port 8000. Daemonized to stay alive while the client runs. |
| `rag_eval.py` | Main script. Loads SQuAD v2, indexes passages in ChromaDB, runs 100 eval queries, computes retrieval and generation metrics, saves results. |
| `launch.py` | Colab launcher. Installs all deps, spawns server.py as detached subprocess, waits for health check, then runs rag_eval.py. |
| `check_progress.py` | Tails log, shows current question index, lists results. |
| `README.md` | Local reference: what this does, how to run, expected results. |

## Architecture

```
[launch.py]
  ├─ pip install vllm chromadb sentence-transformers datasets
  ├─ spawn server.py (detached) ── vLLM API on :8000
  ├─ poll /health until 200
  └─ run rag_eval.py
       ├─ load SQuAD v2 (500 passages, 100 questions)
       ├─ embed passages → ChromaDB index
       ├─ for each question:
       │    ├─ embed question → retrieve top-3 passages
       │    ├─ construct prompt with context
       │    ├─ POST to vLLM /v1/completions
       │    └─ compare answer to ground truth
       ├─ compute metrics
       └─ save /content/rag_results.json
```

## Model & VRAM Budget (T4 16GB)

| Component | VRAM | Notes |
|-----------|------|-------|
| vLLM server (Qwen2.5-7B-AWQ) | ~4.5 GB | `gpu_memory_utilization=0.85` |
| Embedding model | 0 GB (CPU) | all-MiniLM-L6-v2, ~80MB disk |
| ChromaDB index | 0 GB (RAM) | ~200MB for 500 passages |
| **Total GPU** | ~4.5 GB | Well within T4 16GB |

## SQuAD v2 Usage

- Source: `rajpurkar/squad_v2` via `datasets` library
- Passages: first 500 unique context passages from training set
- Questions: first 100 questions from validation set (with non-empty answers)
- Ground truth: `answers.text` field used for Exact Match and F1 scoring

## Metrics

### Retrieval Quality
- **Recall@1**: Fraction of questions where the top-1 retrieved passage contains the answer
- **Recall@3**: Fraction where any of the top-3 passages contains the answer
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of the first relevant passage (higher = better)

### Generation Quality
- **Exact Match**: Fraction of generated answers that contain the ground truth answer string
- **Token F1**: Token-level precision/recall between generated and ground truth answer

### System Performance
- **Avg latency (s/query)**: Wall time per question (embed + retrieve + generate)
- **Tokens/sec**: Generation throughput from vLLM
- **Peak VRAM (GB)**: Max GPU memory during the run

## Prompt Template

```
Answer the question based only on the following context. If the
question cannot be answered from the context, say "unanswerable".

Context:
{passage_text}

Question: {question}
Answer:
```

## Dependencies

Installed in launch.py: `vllm`, `chromadb`, `sentence-transformers`, `datasets`, `requests`, `torch`

## Error Handling

- vLLM server fails to start: exit with diagnostic (GPU check, port conflict)
- Server health check times out (>120s): exit with last server log lines
- Individual query failures: log warning, record error, continue to next question
- Ground truth missing for a question: skip that metric, still record generation output

## Output

`/content/rag_results.json`:
```json
{
  "hardware": {"gpu": "T4", "vram_gb": 16},
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "embedding_model": "all-MiniLM-L6-v2",
    "num_passages": 500,
    "num_questions": 100,
    "retrieval_k": 3
  },
  "retrieval_metrics": {
    "recall_at_1": 0.62,
    "recall_at_3": 0.84,
    "mrr": 0.73
  },
  "generation_metrics": {
    "exact_match": 0.58,
    "token_f1_avg": 0.71
  },
  "system_metrics": {
    "avg_latency_s": 2.3,
    "total_time_s": 230.0,
    "peak_vram_gb": 4.8
  },
  "per_question": [
    {
      "question": "...",
      "ground_truth": "...",
      "generated_answer": "...",
      "retrieved_rank": 1,
      "exact_match": true,
      "token_f1": 0.85,
      "latency_s": 2.1
    }
  ]
}
```

## Constraints

- T4 GPU (16GB VRAM), free-tier Colab session (~2-4h)
- vLLM from pip (prebuilt CUDA 12.x wheels)
- ChromaDB in-memory mode (no persistence needed)
- all-MiniLM-L6-v2 on CPU to keep GPU free for vLLM
- SQuAD v2 loaded via HuggingFace datasets (requires token: `~/.huggingface/access_token`)
- 100 questions should complete in ~4-8 minutes on T4
