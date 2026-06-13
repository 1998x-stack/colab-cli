# RAG: FastText + BM25 + FAISS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid retrieval pipeline (FastText dense + BM25 sparse + FAISS index) evaluated on BeIR/nfcorpus, in cleanrl single-file style, trainable on Colab CPU.

**Architecture:** Single `train.py` with argparse-driven config, sectioned with `──` comments. Three retrieval methods compared: BM25-only, FastText+FAISS-only, and hybrid fusion (weighted sum). Metrics: NDCG@10, MAP@100, Recall@100, MRR. All output to `/content/rag-fasttext-output/`.

**Tech Stack:** fasttext, faiss-cpu, scipy (sparse BM25), numpy, datasets (HF), matplotlib, scikit-learn (CountVectorizer for BM25 tokenization, normalized score fusion)

---

## File Structure

```
projects/rag-fasttext/
├── train.py          # Main implementation (~350 lines, single file)
├── launch.py         # Colab bootstrap (pip install + spawn detached)
└── fetch.sh          # Cron fetch script (3 min interval)
```

Each file has one responsibility:
- `train.py` — data loading, training, indexing, search, eval, visualization
- `launch.py` — dependency installation and subprocess spawning
- `fetch.sh` — remote artifact download via REST

---

### Task 1: Create project directory and train.py skeleton

**Files:**
- Create: `projects/rag-fasttext/train.py`

- [ ] **Step 1: Create directory and write skeleton**

```bash
mkdir -p projects/rag-fasttext
```

- [ ] **Step 2: Write train.py skeleton with imports, argparse, output dirs, and log()**

```python
#!/usr/bin/env python3
"""RAG: FastText + BM25 + FAISS hybrid retrieval on BeIR/nfcorpus.

Three-method comparison:
  1. BM25 (sparse lexical)
  2. FastText + FAISS (dense semantic)
  3. Hybrid fusion (weighted sum)

Metrics: NDCG@10, MAP@100, Recall@100, MRR
"""

import os
import sys
import csv
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sklearn_normalize
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
parser.add_argument("--hybrid_alpha", type=float, default=0.3,
                    help="BM25 weight in hybrid fusion (0=all dense, 1=all sparse)")
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
```

- [ ] **Step 3: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add rag-fasttext train.py skeleton with argparse and logging"
```

---

### Task 2: Data loading — load BeIR/nfcorpus from HuggingFace

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after `log()` function

- [ ] **Step 1: Add data loading section to train.py**

Append to train.py:

```python
# ── Data Loading ─────────────────────────────────────────────────────────────
log(f"Loading dataset: {args.dataset}")

from datasets import load_dataset

# BeIR/nfcorpus has 3 splits: 'corpus' (docs), 'queries' (test queries), 'qrels' (relevance)
corpus_ds = load_dataset(args.dataset, split="corpus", trust_remote_code=False)
queries_ds = load_dataset(args.dataset, split="queries", trust_remote_code=False)
qrels_ds = load_dataset(args.dataset, split="qrels", trust_remote_code=False)

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
query_ids = []
query_texts = []
for row in queries_ds:
    query_ids.append(row["_id"])
    query_texts.append(row["text"])

# Build qrels: query_id → {doc_id: relevance_score}
# nfcorpus uses 0-2 graded relevance
qrels = defaultdict(dict)
for row in qrels_ds:
    qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])

log(f"Corpus: {len(doc_texts)} docs, {sum(len(t.split()) for t in doc_texts) / len(doc_texts):.0f} avg words/doc")
log(f"Queries: {len(query_texts)} queries")
log(f"Qrels: {sum(len(v) for v in qrels.values())} judgments across {len(qrels)} queries")
log(f"Vocabulary size estimate: ~{len(set(w.lower() for t in doc_texts for w in t.split()))} unique tokens")
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add HF data loading for BeIR/nfcorpus"
```

---

### Task 3: Preprocessing — tokenization and FastText training data

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after data loading

- [ ] **Step 1: Add preprocessing section**

Append to train.py:

```python
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add tokenization and FastText training file prep"
```

---

### Task 4: FastText training — unsupervised embeddings on corpus

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after preprocessing

- [ ] **Step 1: Add FastText training section**

Append to train.py:

```python
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add FastText training and document embedding"
```

---

### Task 5: BM25 index — sparse lexical retrieval

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after FastText

- [ ] **Step 1: Add BM25 index and search**

Append to train.py:

```python
# ── BM25 Index ───────────────────────────────────────────────────────────────
log("Building BM25 index ...")
t0 = time.time()

# Use CountVectorizer for term→column mapping
vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",  # match sklearn default
    stop_words=None,                # keep all words for retrieval
)
doc_term_matrix = vectorizer.fit_transform(doc_texts)
vocab = vectorizer.get_feature_names_out()
n_docs, n_terms = doc_term_matrix.shape
log(f"BM25 matrix: {n_docs} docs × {n_terms} terms, {doc_term_matrix.nnz} nonzeros "
    f"({doc_term_matrix.nnz / (n_docs * n_terms) * 100:.2f}% dense)")

# BM25 scoring: score(d,q) = sum(IDF(t) * tf*(t,d) / (k1*(1-b+b*|d|/avgdl) + tf*(t,d)))
# where tf*(t,d) = tf(t,d) and IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
doc_lens = np.array(doc_term_matrix.sum(axis=1)).flatten()
avgdl = doc_lens.mean()
df = np.array((doc_term_matrix > 0).sum(axis=0)).flatten()  # document frequency per term
N = n_docs
idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

# Precompute denominator per doc: k1 * ((1-b) + b * |d|/avgdl)
k1 = args.bm25_k1
b = args.bm25_b
doc_denom = k1 * ((1 - b) + b * doc_lens / avgdl)

log(f"BM25 built in {time.time() - t0:.1f}s, avgdl={avgdl:.1f} words/doc")


def bm25_search(query_text, k=100):
    """Score all documents against query, return top-k (idx, score) pairs."""
    q_vec = vectorizer.transform([query_text])
    # q_vec is (1, n_terms) sparse — each nonzero is a query term
    scores = np.zeros(n_docs)
    cx = q_vec.tocoo()
    for term_idx, tf_q in zip(cx.col, cx.data):
        # Get tf for this term across all docs
        term_col = doc_term_matrix.getcol(term_idx)
        tfs_d = term_col.toarray().flatten()
        term_score = idf[term_idx] * tfs_d * (k1 + 1) / (tfs_d + doc_denom)
        scores += term_score

    top_k = np.argpartition(-scores, min(k, n_docs - 1))[:k]
    top_k = top_k[np.argsort(-scores[top_k])]
    return [(doc_ids[int(idx)], float(scores[idx])) for idx in top_k if scores[idx] > 0]
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add BM25 sparse index with IDF-weighted scoring"
```

---

### Task 6: FAISS index — dense vector search

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after BM25

- [ ] **Step 1: Add FAISS index and search**

Append to train.py:

```python
# ── FAISS Index ──────────────────────────────────────────────────────────────
log(f"Building FAISS HNSW index: M={args.hnsw_M}, ef_construction={args.hnsw_ef_construction} ...")
t0 = time.time()

# L2-normalize for cosine similarity (L2 distance on normalized vectors)
doc_emb_norm = doc_embeddings.copy().astype(np.float32)
faiss.normalize_L2(doc_emb_norm)

dim = args.fasttext_dim
index = faiss.IndexHNSWFlat(dim, args.hnsw_M)
index.hnsw.efConstruction = args.hnsw_ef_construction
index.add(doc_emb_norm)
index.hnsw.efSearch = args.hnsw_ef_search

faiss_build_time = time.time() - t0
log(f"FAISS index built in {faiss_build_time:.1f}s, size={index.ntotal} vectors")


def faiss_search(query_text, k=100):
    """Cosine similarity via FAISS on L2-normalized vectors."""
    words = tokenize(query_text)
    word_vecs = [ft_model.get_word_vector(w) for w in words if w in ft_model]
    if not word_vecs:
        return []
    q = np.mean(word_vecs, axis=0).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, min(k, index.ntotal))
    # L2 distance on normalized vectors → cosine = 1 - d^2/2
    return [(doc_ids[int(indices[0][i])], max(0.0, 1.0 - float(distances[0][i])**2 / 2.0))
            for i in range(len(indices[0])) if indices[0][i] >= 0]
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add FAISS HNSW index with cosine similarity search"
```

---

### Task 7: Hybrid search — weighted fusion of BM25 + FastText

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after FAISS

- [ ] **Step 1: Add hybrid search and score normalization**

Append to train.py:

```python
# ── Hybrid Fusion ────────────────────────────────────────────────────────────
def minmax_normalize(scores):
    """Min-max normalize a dict of {id: score} to [0, 1]."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()))
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return {k: 1.0 for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}


def hybrid_search(query_text, k=100, alpha=None):
    """Weighted fusion: score = alpha * bm25_norm + (1-alpha) * dense_norm."""
    if alpha is None:
        alpha = args.hybrid_alpha

    # Get ranked lists from both methods
    bm25_results = {idx: score for idx, score in bm25_search(query_text, k=k)}
    dense_results = {idx: score for idx, score in faiss_search(query_text, k=k)}

    # Normalize scores
    bm25_norm = minmax_normalize(bm25_results)
    dense_norm = minmax_normalize(dense_results)

    # Fuse
    all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        d = dense_norm.get(doc_id, 0.0)
        fused[doc_id] = alpha * b + (1 - alpha) * d

    # Sort and return top-k
    sorted_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(doc_id, fused[doc_id]) for doc_id in sorted_ids]
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add hybrid fusion search with min-max score normalization"
```

---

### Task 8: Evaluation metrics — NDCG, MAP, Recall, MRR

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after hybrid search

- [ ] **Step 1: Add evaluation metrics**

Append to train.py:

```python
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
        # Get relevance for each retrieved doc (0 if not in qrels)
        rels = np.array([qrels.get(qid, {}).get(str(doc_id), 0) for doc_id, _ in ranked])
        dcg = dcg_at_k(rels, k)
        # Ideal: all relevant docs sorted by relevance
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add IR evaluation metrics (NDCG, MAP, Recall, MRR)"
```

---

### Task 9: Visualization — 4-panel metrics figure

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after evaluation

- [ ] **Step 1: Add visualization function**

Append to train.py:

```python
# ── Visualization ────────────────────────────────────────────────────────────
def plot_results(all_metrics, out_dir):
    """4-panel figure comparing BM25 vs FastText vs Hybrid."""
    methods = [m["method"] for m in all_metrics]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAG Retrieval Comparison — BeIR/nfcorpus", fontsize=14, fontweight="bold")
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

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
    width = 0.25
    for i, m in enumerate(all_metrics):
        vals = [m["ndcg@10"], m["map@100"], m["recall@100"], m["mrr"]]
        bars = ax.bar(x + i * width, vals, width, label=m["method"], color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_title("All Metrics Comparison")
    ax.legend(fontsize=8)
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add 4-panel visualization comparing retrieval methods"
```

---

### Task 10: Main pipeline — CSV metrics and execution orchestration

**Files:**
- Modify: `projects/rag-fasttext/train.py` — append after visualization

- [ ] **Step 1: Add main execution block**

Append to train.py:

```python
# ── Main ─────────────────────────────────────────────────────────────────────
log("=" * 60)
log("RAG: FastText + BM25 + FAISS — Hybrid Retrieval")
log(f"Dataset: {args.dataset}")
log(f"Config: ft_dim={args.fasttext_dim}, ft_epoch={args.fasttext_epoch}, "
    f"bm25_k1={args.bm25_k1}, bm25_b={args.bm25_b}, "
    f"hnsw_M={args.hnsw_M}, hnsw_efc={args.hnsw_ef_construction}, "
    f"hybrid_alpha={args.hybrid_alpha}")
log("=" * 60)

# CSV header
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["method", "ndcg@10", "map@100", "recall@100", "mrr",
                     "latency_ms_mean", "latency_ms_p95", "total_s",
                     "ft_dim", "ft_epoch", "bm25_k1", "bm25_b", "hybrid_alpha"])
csv_file.flush()

all_metrics = []

# Evaluate BM25
bm25_metrics = run_evaluation(lambda q, k: bm25_search(q, k=k), "BM25", k=args.top_k)
all_metrics.append(bm25_metrics)

# Evaluate FastText+FAISS
dense_metrics = run_evaluation(lambda q, k: faiss_search(q, k=k), "FastText+FAISS", k=args.top_k)
all_metrics.append(dense_metrics)

# Evaluate Hybrid
hybrid_metrics = run_evaluation(
    lambda q, k: hybrid_search(q, k=k, alpha=args.hybrid_alpha),
    "Hybrid", k=args.top_k)
all_metrics.append(hybrid_metrics)

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

csv_file.close()
log("Done.")
log(f"Output: {args.out_dir}/")
log(f"  logs/train.log  — full training log")
log(f"  metrics.csv     — evaluation metrics")
log(f"  pngs/retrieval_comparison.png  — comparison figure")
log(f"  index_info.json — index metadata")
```

- [ ] **Step 2: Verify full file syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/train.py
git commit -m "feat: add main pipeline with CSV logging, eval orchestration, and artifact saving"
```

---

### Task 11: launch.py — Colab bootstrap script

**Files:**
- Create: `projects/rag-fasttext/launch.py`

- [ ] **Step 1: Write launch.py**

```python
#!/usr/bin/env python3
"""Launch RAG FastText training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["fasttext", "faiss-cpu", "scipy", "numpy", "datasets", "matplotlib", "scikit-learn"]
SCRIPT = "train.py"
LOG = "/content/rag-fasttext-output/logs/train.log"

print("=== Colab RAG: FastText + BM25 + FAISS Launcher ===")
print(f"Installing: {DEPS}")
for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/rag-fasttext-output/logs", exist_ok=True)
os.makedirs("/content/rag-fasttext-output/pngs", exist_ok=True)
os.makedirs("/content/rag-fasttext-output/checkpoints", exist_ok=True)

print(f"\nLaunching {SCRIPT} detached ...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
```

- [ ] **Step 2: Verify syntax**

```bash
cd projects/rag-fasttext && python3 -c "import ast; ast.parse(open('launch.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/launch.py
git commit -m "feat: add colab launch.py bootstrap for rag-fasttext"
```

---

### Task 12: fetch.sh — cron download script

**Files:**
- Create: `projects/rag-fasttext/fetch.sh`

- [ ] **Step 1: Write fetch.sh**

```bash
#!/bin/bash
# Fetch RAG FastText training results from Colab VM. Called by cron every 3 minutes.
set -euo pipefail
SESSION="${1:-rag-fasttext}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="rag-fasttext-output.tar.gz"
mkdir -p "$LOCAL_OUT"

export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890
COLAB="$(which colab)"

echo "[fetch] $(date '+%H:%M:%S') Starting fetch for session: $SESSION"

# 1. Check session alive
echo "[fetch] Checking session ..."
"$COLAB" sessions 2>/dev/null | grep -q "$SESSION" || {
    echo "[fetch] WARNING: session $SESSION not found — may be dead"
    exit 0
}

# 2. Tar output on VM (exclude checkpoints to keep tar small)
echo "[fetch] Tarring output on VM ..."
echo '
import subprocess, os
out = "/content/rag-fasttext-output"
tar = "/content/rag-fasttext-output.tar.gz"
# Exclude checkpoints and fasttext_train.txt
subprocess.run(["tar", "-czf", tar, "-C", out,
    "--exclude=checkpoints", "--exclude=fasttext_train.txt",
    "--exclude=*.bin", "--exclude=*.pt",
    ".", "-C", "/content",
    "--transform=s|^./||g"], check=True)
print(f"Tarball: {os.path.getsize(tar)/1024:.0f} KB")
' | "$COLAB" exec -s "$SESSION" --timeout 15 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed, trying direct download ..."
}

# 3. Download
echo "[fetch] Downloading ..."
"$COLAB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or output not ready"
    exit 0
}

# 4. Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" 2>/dev/null || { echo "[fetch] WARNING: extract failed"; exit 0; }
echo "[fetch] $(date '+%H:%M:%S') Done."

# 5. Report
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""; echo "══ Last 10 log lines ══"
    tail -10 "$LOCAL_OUT/logs/train.log"
fi
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""; echo "══ Metrics ══"
    cat "$LOCAL_OUT/metrics.csv"
fi
if [ -f "$LOCAL_OUT/index_info.json" ]; then
    echo ""; echo "══ Index Info ══"
    python3 -c "import json; d=json.load(open('$LOCAL_OUT/index_info.json')); print(json.dumps(d.get('metrics',{}), indent=2))" 2>/dev/null || true
fi
echo ""; echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"
echo ""; echo "Files in: $LOCAL_OUT"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x projects/rag-fasttext/fetch.sh
```

- [ ] **Step 3: Commit**

```bash
git add projects/rag-fasttext/fetch.sh
git commit -m "feat: add fetch.sh cron script for 3-min Colab artifact sync"
```

---

### Task 13: Local dry-run test

**Files:**
- None (verification only)

- [ ] **Step 1: Install dependencies locally**

```bash
pip install fasttext faiss-cpu scipy numpy datasets matplotlib scikit-learn -q
```

- [ ] **Step 2: Run train.py locally with small config to verify pipeline works end-to-end**

```bash
cd projects/rag-fasttext && python3 train.py --fasttext_epoch 3 --out_dir /tmp/rag-test
```

Expected: Full pipeline completes. Check output:
- `/tmp/rag-test/logs/train.log` exists with timestamps
- `/tmp/rag-test/metrics.csv` has 3 rows (BM25, FastText+FAISS, Hybrid)
- `/tmp/rag-test/pngs/retrieval_comparison.png` exists
- `/tmp/rag-test/index_info.json` exists
- NDCG@10 values are between 0.0 and 1.0

- [ ] **Step 3: Verify expected output**

```bash
echo "=== Log ===" && cat /tmp/rag-test/logs/train.log | head -5
echo "=== Metrics ===" && cat /tmp/rag-test/metrics.csv
echo "=== PNGs ===" && ls -lh /tmp/rag-test/pngs/
echo "=== Index info ===" && python3 -c "import json; print(json.dumps(json.load(open('/tmp/rag-test/index_info.json')), indent=2))" | head -20
```

- [ ] **Step 4: Commit if any fixes applied**

```bash
git add projects/rag-fasttext/train.py
git commit -m "fix: local dry-run fixes" || echo "No fixes needed"
```
