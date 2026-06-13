#!/usr/bin/env python3
"""RAG: FastText + BM25 + FAISS hybrid retrieval on BeIR/nfcorpus.

Three-method comparison:
  1. BM25 (sparse lexical)
  2. FastText + FAISS (dense semantic)
  3. Hybrid fusion (weighted sum)

Metrics: NDCG@10, MAP@100, Recall@100, MRR
"""

import os
import csv
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import fasttext
import faiss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="BeIR/nfcorpus")
parser.add_argument("--fasttext_dim", type=int, default=100)
parser.add_argument("--fasttext_epoch", type=int, default=10)
parser.add_argument("--bm25_k1", type=float, default=1.5)
parser.add_argument("--bm25_b", type=float, default=0.75)
parser.add_argument("--hnsw_M", type=int, default=16)
parser.add_argument("--hnsw_ef_construction", type=int, default=200)
parser.add_argument("--hnsw_ef_search", type=int, default=64)
parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                    help="BM25 weight in weighted-score fusion (0=all dense, 1=all sparse)")
parser.add_argument("--top_k", type=int, default=100)
parser.add_argument("--out_dir", default="/content/rag-fasttext-output")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/logs", exist_ok=True)
os.makedirs(f"{args.out_dir}/pngs", exist_ok=True)
os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)

LOG_PATH = f"{args.out_dir}/logs/train.log"
CSV_PATH = f"{args.out_dir}/metrics.csv"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── Data Loading ─────────────────────────────────────────────────────────────
log(f"Loading dataset: {args.dataset}")

from datasets import load_dataset

# BeIR/nfcorpus uses configs (not splits): 'corpus' for docs, 'queries' for queries
# Qrels are in a separate dataset: BeIR/nfcorpus-qrels
corpus_ds = load_dataset(args.dataset, "corpus", split="corpus", trust_remote_code=False)
queries_ds = load_dataset(args.dataset, "queries", split="queries", trust_remote_code=False)
qrels_ds = load_dataset(f"{args.dataset}-qrels", split="test", trust_remote_code=False)

# Build doc lookup: _id → text
# nfcorpus fields: _id, title, text
doc_ids = []
doc_texts = []
for row in corpus_ds:
    doc_ids.append(row["_id"])
    # Combine title + text for richer representation
    title = row.get("title", "")
    body = row.get("text", "")
    doc_texts.append(f"{title} {body}".strip())

# Build query lookup: _id → text
# nfcorpus queries have title='' and text field (only text is meaningful)
query_ids = []
query_texts = []
for row in queries_ds:
    query_ids.append(row["_id"])
    query_texts.append(row["text"])

# Build qrels: query_id → {doc_id: relevance_score}
# nfcorpus-qrels uses 0-2 graded relevance
qrels = defaultdict(dict)
for row in qrels_ds:
    qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])

log(f"Corpus: {len(doc_texts)} docs, {sum(len(t.split()) for t in doc_texts) / len(doc_texts):.0f} avg words/doc")
log(f"Queries: {len(query_texts)} queries")
log(f"Qrels: {sum(len(v) for v in qrels.values())} judgments across {len(qrels)} queries")
log(f"Vocabulary size estimate: ~{len(set(w.lower() for t in doc_texts for w in t.split()))} unique tokens")


# ── Preprocessing ────────────────────────────────────────────────────────────
log("Preprocessing: tokenization and FastText training file ...")

# Simple whitespace tokenization (no NLTK/spaCy to keep deps minimal)
def tokenize(text):
    return text.lower().split()

# Tokenize all docs for BM25 vocabulary
tokenized_docs = [tokenize(d) for d in doc_texts]
tokenized_queries = [tokenize(q) for q in query_texts]

# FastText expects a text file with one sentence per line
ft_train_path = f"{args.out_dir}/fasttext_train.txt"
with open(ft_train_path, "w") as f:
    for doc in doc_texts:
        # FastText reads one "sentence" per line
        f.write(doc.lower().replace("\n", " ") + "\n")

log(f"FastText training file: {ft_train_path} ({os.path.getsize(ft_train_path) / 1024:.1f} KB)")


# ── FastText Training ────────────────────────────────────────────────────────
log(f"Training FastText: dim={args.fasttext_dim}, epoch={args.fasttext_epoch} ...")
t0 = time.time()

ft_model = fasttext.train_unsupervised(
    ft_train_path,
    model="skipgram",
    dim=args.fasttext_dim,
    epoch=args.fasttext_epoch,
    minCount=1,          # small corpus: keep all words
    minn=3, maxn=6,      # character n-grams for OOV
    thread=os.cpu_count() or 2,
    verbose=0,
)

ft_train_time = time.time() - t0
log(f"FastText trained in {ft_train_time:.1f}s, vocab={len(ft_model.words)} words")

# Document embeddings: mean of word vectors
def embed_docs(texts):
    """Mean-pool FastText word vectors → (n_docs, dim) matrix."""
    vecs = np.zeros((len(texts), args.fasttext_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = tokenize(text)
        word_vecs = [ft_model.get_word_vector(w) for w in words if w in ft_model]
        if word_vecs:
            vecs[i] = np.mean(word_vecs, axis=0)
        # else: stays zero (document with all OOV words)
    return vecs

t0 = time.time()
doc_embeddings = embed_docs(doc_texts)
query_embeddings = embed_docs(query_texts)
log(f"Embedding time: {time.time() - t0:.1f}s (docs={doc_embeddings.shape}, queries={query_embeddings.shape})")


# ── BM25 Index ───────────────────────────────────────────────────────────────
log("Building BM25 index ...")
t0 = time.time()

# Use CountVectorizer for term→column mapping
vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    stop_words=None,                # keep all words for retrieval
)
doc_term_matrix = vectorizer.fit_transform(doc_texts)
vocab = vectorizer.get_feature_names_out()
n_docs, n_terms = doc_term_matrix.shape
log(f"BM25 matrix: {n_docs} docs x {n_terms} terms, {doc_term_matrix.nnz} nonzeros "
    f"({doc_term_matrix.nnz / (n_docs * n_terms) * 100:.2f}% dense)")

# BM25 scoring
doc_lens = np.array(doc_term_matrix.sum(axis=1)).flatten()
avgdl = doc_lens.mean()
df = np.array((doc_term_matrix > 0).sum(axis=0)).flatten()
N = n_docs
idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

k1 = args.bm25_k1
b = args.bm25_b
doc_denom = k1 * ((1 - b) + b * doc_lens / avgdl)

log(f"BM25 built in {time.time() - t0:.1f}s, avgdl={avgdl:.1f} words/doc")


def bm25_search(query_text, k=100):
    """Score all documents against query, return top-k (doc_id, score) pairs."""
    q_vec = vectorizer.transform([query_text])
    scores = np.zeros(n_docs)
    cx = q_vec.tocoo()
    for term_idx, tf_q in zip(cx.col, cx.data):
        term_col = doc_term_matrix.getcol(term_idx)
        tfs_d = term_col.toarray().flatten()
        term_score = idf[term_idx] * tfs_d * (k1 + 1) / (tfs_d + doc_denom)
        scores += term_score

    top_k = np.argpartition(-scores, min(k, n_docs - 1))[:k]
    top_k = top_k[np.argsort(-scores[top_k])]
    return [(doc_ids[int(idx)], float(scores[idx])) for idx in top_k if scores[idx] > 0]


# ── FAISS Index ──────────────────────────────────────────────────────────────
log("Building FAISS index (IndexFlatIP for cosine similarity) ...")
t0 = time.time()

# L2-normalize for cosine similarity via inner product
doc_emb_norm = np.ascontiguousarray(doc_embeddings.astype(np.float32))
faiss.normalize_L2(doc_emb_norm)

dim = args.fasttext_dim
index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vectors = cosine similarity
index.add(doc_emb_norm)

faiss_build_time = time.time() - t0
log(f"FAISS index built in {faiss_build_time:.1f}s, size={index.ntotal} vectors")


def faiss_search(query_text, k=100):
    """Cosine similarity via FAISS inner product on L2-normalized vectors."""
    words = tokenize(query_text)
    word_vecs = [ft_model.get_word_vector(w) for w in words if w in ft_model]
    if not word_vecs:
        return []
    q = np.ascontiguousarray(np.mean(word_vecs, axis=0).astype(np.float32).reshape(1, -1))
    faiss.normalize_L2(q)
    distances, indices = index.search(q, min(k, index.ntotal))
    # distances are inner product (cosine similarity) since vectors are L2-normalized
    return [(doc_ids[int(indices[0][i])], max(0.0, float(distances[0][i])))
            for i in range(len(indices[0])) if indices[0][i] >= 0]


# ── Hybrid Fusion ────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(ranked_lists, k=60):
    """RRF: score(d) = sum_i 1/(k + rank_i(d)) — no normalization needed."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [(doc_id, scores[doc_id]) for doc_id in sorted_ids]


def weighted_score_fusion(query_text, k=100, alpha=None):
    """Weighted sum with min-max normalization (original method)."""
    if alpha is None:
        alpha = args.hybrid_alpha
    bm25_results = {idx: score for idx, score in bm25_search(query_text, k=k)}
    dense_results = {idx: score for idx, score in faiss_search(query_text, k=k)}

    # Min-max normalize
    def normalize(scores):
        if not scores:
            return {}
        vals = np.array(list(scores.values()))
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            return {k: 1.0 for k in scores}
        return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}

    bm25_norm = normalize(bm25_results)
    dense_norm = normalize(dense_results)
    all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        d = dense_norm.get(doc_id, 0.0)
        fused[doc_id] = alpha * b + (1 - alpha) * d
    sorted_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(doc_id, fused[doc_id]) for doc_id in sorted_ids]


def hybrid_search(query_text, k=100, alpha=None):
    """RRF fusion (primary method) — robust, no score normalization."""
    bm25_ranked = bm25_search(query_text, k=k)
    dense_ranked = faiss_search(query_text, k=k)
    return reciprocal_rank_fusion([bm25_ranked, dense_ranked], k=60)


# ── Evaluation Metrics ───────────────────────────────────────────────────────
def dcg_at_k(rels, k):
    """Discounted Cumulative Gain at k."""
    rels = rels[:k]
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return np.sum(rels / discounts)


def ndcg_at_k(results, qrels, k=10):
    """Normalized DCG@k averaged over queries."""
    scores = []
    for qid, ranked in results.items():
        rels = np.array([qrels.get(qid, {}).get(str(doc_id), 0) for doc_id, _ in ranked])
        dcg = dcg_at_k(rels, k)
        ideal_rels = np.sort([r for r in qrels.get(qid, {}).values() if r > 0])[::-1]
        idcg = dcg_at_k(ideal_rels, k) if len(ideal_rels) > 0 else 1.0
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(scores)


def average_precision(ranked, relevant_docs, k=100):
    """Average Precision at k."""
    ranked = ranked[:k]
    num_hits = 0
    sum_precisions = 0.0
    num_relevant = len(relevant_docs)
    if num_relevant == 0:
        return 0.0
    for i, (doc_id, _) in enumerate(ranked):
        if str(doc_id) in relevant_docs:
            num_hits += 1
            sum_precisions += num_hits / (i + 1)
    return sum_precisions / min(num_relevant, k)


def map_at_k(results, qrels, k=100):
    """Mean Average Precision at k."""
    scores = []
    for qid, ranked in results.items():
        relevant = {str(did): score for did, score in qrels.get(qid, {}).items() if score > 0}
        scores.append(average_precision(ranked, relevant, k))
    return np.mean(scores)


def recall_at_k(results, qrels, k=100):
    """Recall@k: fraction of relevant docs retrieved in top-k."""
    scores = []
    for qid, ranked in results.items():
        ranked_ids = {str(doc_id) for doc_id, _ in ranked[:k]}
        relevant = {str(did) for did, score in qrels.get(qid, {}).items() if score > 0}
        if len(relevant) == 0:
            continue
        scores.append(len(ranked_ids & relevant) / len(relevant))
    return np.mean(scores)


def mrr(results, qrels):
    """Mean Reciprocal Rank: 1/rank of first relevant document."""
    scores = []
    for qid, ranked in results.items():
        relevant = {str(did) for did, score in qrels.get(qid, {}).items() if score > 0}
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            if str(doc_id) in relevant:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return np.mean(scores)


def run_evaluation(search_fn, method_name, k=100):
    """Run search_fn on all queries, return metrics dict and per-query latencies."""
    log(f"Evaluating {method_name} ...")
    t0 = time.time()
    results = {}
    latencies = []
    for qid, qtext in zip(query_ids, query_texts):
        t1 = time.time()
        results[qid] = search_fn(qtext, k=k)
        latencies.append((time.time() - t1) * 1000)  # ms

    metrics = {
        "method": method_name,
        "ndcg@10": ndcg_at_k(results, qrels, k=10),
        "map@100": map_at_k(results, qrels, k=100),
        "recall@100": recall_at_k(results, qrels, k=100),
        "mrr": mrr(results, qrels),
        "latency_ms_mean": np.mean(latencies),
        "latency_ms_p95": np.percentile(latencies, 95),
        "total_s": time.time() - t0,
    }
    log(f"  {method_name}: NDCG@10={metrics['ndcg@10']:.4f}  MAP@100={metrics['map@100']:.4f}  "
        f"Recall@100={metrics['recall@100']:.4f}  MRR={metrics['mrr']:.4f}  "
        f"lat={metrics['latency_ms_mean']:.1f}ms/q")
    return metrics


# ── Visualization ────────────────────────────────────────────────────────────
def plot_results(all_metrics, out_dir):
    """4-panel figure comparing BM25 vs FastText vs Hybrid."""
    methods = [m["method"] for m in all_metrics]
    n = len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(max(14, n * 2), 12))
    fig.suptitle("RAG Retrieval Comparison -- BeIR/nfcorpus", fontsize=14, fontweight="bold")
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
              "#1abc9c", "#f39c12", "#2980b9"][:n]

    # Panel 1: NDCG@10 bars
    ax = axes[0, 0]
    vals = [m["ndcg@10"] for m in all_metrics]
    bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("NDCG@10 (higher is better)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Recall@100 bars
    ax = axes[0, 1]
    vals = [m["recall@100"] for m in all_metrics]
    bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Recall@100 (higher is better)")
    ax.set_ylabel("Recall@100")
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Multi-metric grouped bars
    ax = axes[1, 0]
    metric_names = ["NDCG@10", "MAP@100", "Recall@100", "MRR"]
    x = np.arange(len(metric_names))
    width = min(0.25, 0.8 / n)  # adaptive width
    for i, m in enumerate(all_metrics):
        vals = [m["ndcg@10"], m["map@100"], m["recall@100"], m["mrr"]]
        bars = ax.bar(x + i * width, vals, width, label=m["method"], color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_title("All Metrics Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Latency comparison
    ax = axes[1, 1]
    method_lat = [m["latency_ms_mean"] for m in all_metrics]
    bars = ax.bar(methods, method_lat, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Mean Query Latency (ms, lower is better)")
    ax.set_ylabel("Latency (ms)")
    for bar, v in zip(bars, method_lat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/pngs/retrieval_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"Figure saved: {out_dir}/pngs/retrieval_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────
log("=" * 60)
log("RAG: FastText + BM25 + FAISS -- Hybrid Retrieval")
log(f"Dataset: {args.dataset}")
log(f"Config: ft_dim={args.fasttext_dim}, ft_epoch={args.fasttext_epoch}, "
    f"bm25_k1={args.bm25_k1}, bm25_b={args.bm25_b}, "
    f"hnsw_M={args.hnsw_M}, hnsw_efc={args.hnsw_ef_construction}, "
    f"hybrid_alpha={args.hybrid_alpha}")
log("=" * 60)

# CSV header
with open(CSV_PATH, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["method", "ndcg@10", "map@100", "recall@100", "mrr",
                         "latency_ms_mean", "latency_ms_p95", "total_s",
                         "ft_dim", "ft_epoch", "bm25_k1", "bm25_b", "hybrid_alpha"])
    csv_file.flush()

    all_metrics = []

    # Evaluate BM25
    bm25_metrics = run_evaluation(bm25_search, "BM25", k=args.top_k)
    all_metrics.append(bm25_metrics)

    # Evaluate FastText+FAISS
    dense_metrics = run_evaluation(faiss_search, "FastText+FAISS", k=args.top_k)
    all_metrics.append(dense_metrics)

    # Evaluate RRF (no alpha needed — parameter-free)
    rrf_metrics = run_evaluation(
        lambda q, k: hybrid_search(q, k=k), "Hybrid-RRF", k=args.top_k)
    all_metrics.append(rrf_metrics)

    # Sweep weighted-fusion alphas to find best
    best_alpha_metrics = None
    best_alpha_ndcg = 0
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        m = run_evaluation(
            lambda q, k, a=alpha: weighted_score_fusion(q, k=k, alpha=a),
            f"Hybrid-α={alpha:.1f}", k=args.top_k)
        all_metrics.append(m)
        if m["ndcg@10"] > best_alpha_ndcg:
            best_alpha_ndcg = m["ndcg@10"]
            best_alpha_metrics = m
    log(f"Best weighted alpha: {best_alpha_metrics['method']} (NDCG@10={best_alpha_ndcg:.4f})")

    # Write CSV rows
    for m in all_metrics:
        csv_writer.writerow([
            m["method"], m["ndcg@10"], m["map@100"], m["recall@100"], m["mrr"],
            m["latency_ms_mean"], m["latency_ms_p95"], m["total_s"],
            args.fasttext_dim, args.fasttext_epoch, args.bm25_k1, args.bm25_b,
            args.hybrid_alpha,
        ])
        csv_file.flush()

# Generate visualization
plot_results(all_metrics, args.out_dir)

# Save index metadata
index_info = {
    "dataset": args.dataset,
    "n_docs": n_docs,
    "n_queries": len(query_texts),
    "fasttext_dim": args.fasttext_dim,
    "fasttext_epoch": args.fasttext_epoch,
    "fasttext_train_s": ft_train_time,
    "faiss_build_s": faiss_build_time,
    "faiss_index_size": index.ntotal,
    "bm25_vocab_size": int(n_terms),
    "metrics": {m["method"]: {k: v for k, v in m.items() if k != "method"} for m in all_metrics},
}
with open(f"{args.out_dir}/index_info.json", "w") as f:
    json.dump(index_info, f, indent=2)

# Save FAISS index for reuse
faiss.write_index(index, f"{args.out_dir}/checkpoints/faiss_index.bin")
log(f"FAISS index saved: {args.out_dir}/checkpoints/faiss_index.bin")

# Save FastText model
ft_model.save_model(f"{args.out_dir}/checkpoints/fasttext_model.bin")
log(f"FastText model saved: {args.out_dir}/checkpoints/fasttext_model.bin")

log("Done.")
log(f"Output: {args.out_dir}/")
log("  logs/train.log  -- full training log")
log("  metrics.csv     -- evaluation metrics")
log("  pngs/retrieval_comparison.png  -- comparison figure")
log("  index_info.json -- index metadata")
