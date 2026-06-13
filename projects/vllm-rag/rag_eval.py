"""RAG Evaluation Pipeline — SQuAD v2 + vLLM + ChromaDB.

Loads SQuAD v2, indexes passages in ChromaDB with sentence-transformers
embeddings, runs 100 questions through retrieve→generate, and computes
retrieval + generation quality metrics.

vLLM server must be running on http://localhost:8000 before this script.
"""
import json
import os
import sys
import time
import logging

import requests
import torch

logging.getLogger("chromadb").setLevel(logging.WARNING)

# ── Configuration ──────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VLLM_URL = "http://localhost:8000/v1/completions"
NUM_PASSAGES = 500
NUM_QUESTIONS = 100
RETRIEVAL_K = 3
MAX_TOKENS = 128
TEMPERATURE = 0.0
OUTPUT_FILE = "/content/rag_results.json"

PROMPT_TEMPLATE = (
    "Answer the question based only on the following context. "
    "If the question cannot be answered from the context, say 'unanswerable'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_hf_token() -> str | None:
    """Read HuggingFace token from ~/.huggingface/access_token or env."""
    token_path = os.path.expanduser("~/.huggingface/access_token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_squad() -> tuple[list[str], list[dict]]:
    """Load SQuAD v2, returning (passages, questions).

    Passages: first NUM_PASSAGES unique context strings from training set.
    Questions: first NUM_QUESTIONS from validation set with non-empty answers.
    """
    log("Loading SQuAD v2 dataset...")
    from datasets import load_dataset

    # Auth token for gated datasets (SQuAD v2 is public, included for pattern)
    token = read_hf_token()
    kw = {"token": token} if token else {}

    squad = load_dataset("rajpurkar/squad_v2", **kw)

    # Extract unique passages from training set
    passages: list[str] = []
    seen: set[str] = set()
    for row in squad["train"]:
        ctx = row["context"].strip()
        if ctx and ctx not in seen:
            passages.append(ctx)
            seen.add(ctx)
            if len(passages) >= NUM_PASSAGES:
                break

    log(f"  Extracted {len(passages)} unique passages")

    # Extract questions with answers from validation set
    questions: list[dict] = []
    for row in squad["validation"]:
        if len(row["answers"]["text"]) > 0:
            questions.append({
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answers": row["answers"]["text"],
            })
            if len(questions) >= NUM_QUESTIONS:
                break

    log(f"  Extracted {len(questions)} questions with answers")
    return passages, questions


def build_index(passages: list[str]) -> "chromadb.Collection":  # noqa: F821
    """Embed passages with sentence-transformers and index in ChromaDB."""
    log(f"Building ChromaDB index with {len(passages)} passages...")
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="squad_passages",
        embedding_function=ef,
    )

    # Add in batches to avoid overwhelming the embedding model
    batch_size = 100
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        ids = [f"p{j}" for j in range(i, i + len(batch))]
        collection.add(documents=batch, ids=ids)

    log(f"  Indexed {collection.count()} passages")
    return collection


def retrieve(collection, question: str, k: int = RETRIEVAL_K) -> list[str]:
    """Retrieve top-k passages for a question. Returns list of passage texts."""
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    return [d for d in docs if d]


def generate(prompt: str) -> tuple[str, float]:
    """Send prompt to vLLM API, return (answer_text, generation_time_s)."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.time()
    resp = requests.post(VLLM_URL, json=payload, timeout=60)
    gen_time = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["text"].strip()
    return answer, gen_time


def answer_in_text(answer: str, text: str) -> bool:
    """Check if answer string appears in text (case-insensitive)."""
    return answer.lower().strip() in text.lower()


def token_f1(pred: str, refs: list[str]) -> float:
    """Token-level F1 between predicted answer and reference answers.

    Uses whitespace tokenization (lowercased). Returns best F1 across
    all reference answers (standard SQuAD evaluation practice).
    """
    pred_tokens = set(pred.lower().split())
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in refs:
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def compute_metrics(per_q: list[dict]) -> dict:
    """Aggregate per-question results into summary metrics."""
    n = len(per_q)

    # Retrieval metrics
    recall1 = sum(1 for q in per_q if q.get("retrieved_rank") == 1) / n
    recall3 = sum(1 for q in per_q if q.get("retrieved_rank") is not None and q["retrieved_rank"] <= RETRIEVAL_K) / n
    mrr = sum(
        (1.0 / q["retrieved_rank"]) if q.get("retrieved_rank") else 0.0
        for q in per_q
    ) / n

    # Generation metrics
    em = sum(1 for q in per_q if q.get("exact_match")) / n
    f1_avg = sum(q.get("token_f1", 0.0) for q in per_q) / n

    # System metrics
    latencies = [q["latency_s"] for q in per_q if q.get("latency_s")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_time = sum(latencies)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    return {
        "retrieval_metrics": {
            "recall_at_1": round(recall1, 4),
            "recall_at_3": round(recall3, 4),
            "mrr": round(mrr, 4),
        },
        "generation_metrics": {
            "exact_match": round(em, 4),
            "token_f1_avg": round(f1_avg, 4),
        },
        "system_metrics": {
            "avg_latency_s": round(avg_latency, 2),
            "total_time_s": round(total_time, 1),
            "peak_vram_gb": round(peak_vram, 2),
        },
    }


def run_eval(passages: list[str], questions: list[dict]) -> tuple[list[dict], dict]:
    """Run the full RAG evaluation pipeline.

    Returns (per_question_results, aggregate_metrics).
    """
    collection = build_index(passages)
    torch.cuda.reset_peak_memory_stats()

    per_question: list[dict] = []
    log(f"\nRunning {len(questions)} evaluation queries...\n")

    for i, q in enumerate(questions):
        q_start = time.time()

        # Retrieve
        retrieved = retrieve(collection, q["question"], k=RETRIEVAL_K)

        # Find rank of first relevant passage
        rank = None
        for r_idx, passage in enumerate(retrieved):
            for ans in q["answers"]:
                if answer_in_text(ans, passage):
                    rank = r_idx + 1
                    break
            if rank is not None:
                break

        # Generate
        context = "\n\n".join(retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, question=q["question"])
        try:
            answer, gen_time = generate(prompt)
        except Exception as exc:
            log(f"  [{i+1}/{len(questions)}] ERROR: {exc}")
            answer = ""

        # Score
        em = any(answer_in_text(a, answer) for a in q["answers"]) if answer else False
        f1 = token_f1(answer, q["answers"]) if answer else 0.0
        latency = time.time() - q_start

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["answers"],
            "generated_answer": answer,
            "retrieved_rank": rank,
            "exact_match": em,
            "token_f1": round(f1, 4),
            "latency_s": round(latency, 2),
        })

        if (i + 1) % 10 == 0:
            log(
                f"  [{i+1}/{len(questions)}] "
                f"EM={sum(1 for x in per_question if x['exact_match'])}/{i+1}  "
                f"avg_lat={sum(x['latency_s'] for x in per_question)/(i+1):.1f}s"
            )

    metrics = compute_metrics(per_question)
    return per_question, metrics


def main() -> None:
    log("=== RAG Pipeline Evaluation ===")
    log(f"Model: Qwen/Qwen2.5-7B-Instruct-AWQ (vLLM @ {VLLM_URL})")
    log(f"Embedding: {EMBEDDING_MODEL}")
    log(f"Passages: {NUM_PASSAGES} | Questions: {NUM_QUESTIONS} | K: {RETRIEVAL_K}")

    # Wait for vLLM server to be ready
    log("Waiting for vLLM server to be ready...")
    deadline = time.time() + 300  # 5-minute timeout for model download + load
    while time.time() < deadline:
        try:
            resp = requests.get("http://localhost:8000/health", timeout=5)
            if resp.status_code == 200:
                log("vLLM server is ready!")
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        log("ERROR: vLLM server failed to start within 5 minutes.")
        sys.exit(1)

    passages, questions = load_squad()
    per_question, metrics = run_eval(passages, questions)

    # Print summary
    log(f"\n{'=' * 50}")
    log("RESULTS")
    log(f"{'=' * 50}")

    rm = metrics["retrieval_metrics"]
    gm = metrics["generation_metrics"]
    sm = metrics["system_metrics"]

    print("\n── Retrieval ──")
    print(f"  Recall@1:  {rm['recall_at_1']:.3f}")
    print(f"  Recall@3:  {rm['recall_at_3']:.3f}")
    print(f"  MRR:       {rm['mrr']:.3f}")

    print("\n── Generation ──")
    print(f"  Exact Match:  {gm['exact_match']:.3f}")
    print(f"  Token F1:     {gm['token_f1_avg']:.3f}")

    print("\n── System ──")
    print(f"  Avg latency:  {sm['avg_latency_s']:.2f} s/query")
    print(f"  Total time:   {sm['total_time_s']:.0f} s")
    print(f"  Peak VRAM:    {sm['peak_vram_gb']:.2f} GB")

    # Save results
    output = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
            ) if torch.cuda.is_available() else 0,
        },
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "embedding_model": EMBEDDING_MODEL,
            "num_passages": NUM_PASSAGES,
            "num_questions": NUM_QUESTIONS,
            "retrieval_k": RETRIEVAL_K,
        },
        **metrics,
        "per_question": per_question,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {OUTPUT_FILE}")

    # Takeaways
    print("\n── Key Takeaways ──")
    print(
        "• vLLM's OpenAI-compatible API lets any HTTP client use a locally-served LLM\n"
        "  — the same code works against OpenAI, vLLM, and other compatible servers."
    )
    print(
        "• Retrieval quality (Recall@k) is the ceiling for RAG performance —\n"
        "  if the right passage isn't retrieved, the LLM can't answer correctly."
    )
    print(
        "• ChromaDB with sentence-transformers provides a zero-config vector store\n"
        "  that fits entirely in memory for small-to-medium document sets."
    )
    print(
        "• Running both the server and client on a single T4 is practical for\n"
        "  development and evaluation — production would split these across GPUs."
    )


if __name__ == "__main__":
    main()
