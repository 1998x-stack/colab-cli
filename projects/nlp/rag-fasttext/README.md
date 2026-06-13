# RAG-FastText

Hybrid retrieval-augmented generation pipeline comparing BM25 (sparse lexical), FastText + FAISS (dense semantic), and weighted fusion on the BeIR/nfcorpus dataset.

## Usage

```bash
# Local training (runs all three methods + evaluation)
python train.py

# Colab deployment (see ../.claude/skills/colab-cli/SKILL.md)
cb launch.py
```

## Key results

| Metric | BM25 | FastText+FAISS | Hybrid (alpha=0.3) |
|--------|------|----------------|---------------------|
| NDCG@10 | 0.0310 | 0.0134 | 0.0190 |
| MAP@100 | 0.0149 | 0.0053 | 0.0086 |
| Recall@100 | 0.2364 | 0.1690 | **0.2489** |
| MRR | 0.0523 | 0.0275 | 0.0360 |
| Latency (ms) | 2.23 | 0.51 | 3.16 |

BM25 achieves the best NDCG@10 and MRR. Hybrid fusion achieves the best Recall@100. FastText+FAISS is fastest but weakest on all accuracy metrics. The nfcorpus dataset is small (3,633 docs, 3,237 queries) with graded relevance (0-2).

## Gotchas

- The FastText model is trained on the corpus itself (unsupervised skip-gram, 100 dim, 10 epochs), not on a pretrained embedding -- quality is limited by the small nfcorpus corpus.
- Document embeddings are mean-pooled word vectors; documents with all OOV words get zero vectors and cannot be retrieved via FAISS.
- FAISS uses IndexFlatIP (brute-force inner product) on L2-normalized vectors for cosine similarity. Exact search is fine for 3.6K docs but would not scale to millions.
- BM25 is implemented from scratch using CountVectorizer sparse matrices (not using rank_bm25 or Elasticsearch) to keep dependencies minimal.
- BM25 latency (2.23 ms/query) is dominated by sparse matrix operations; FAISS (0.51 ms) is faster due to optimized BLAS.
- Hybrid scores are min-max normalized per method before weighted fusion, which can distort scores when few documents score above zero.
- The dataset uses config-based loading: `corpus` and `queries` configs, not train/test splits. Qrels are in a separate `-qrels` dataset.
- Character n-grams (minn=3, maxn=6) help FastText handle OOV words at query time.
