# RAG: FastText + BM25 + FAISS — Design Spec

**Date:** 2026-06-13
**Dataset:** BeIR/nfcorpus (3,633 docs, 323 queries, graded qrels)
**Training:** Colab CPU, ~1-2 min total

## Architecture

Three-stage retrieval pipeline with hybrid fusion:

1. **FastText** (100d, skipgram) — unsupervised training on document corpus. Doc embedding = mean pooled token vectors. Subword n-grams handle OOV.
2. **BM25** — scipy sparse CSR implementation. k1=1.5, b=0.75. No training, corpus statistics only.
3. **FAISS** — IndexHNSWFlat (M=16, ef_construction=200). Approximate nearest neighbor on FastText embeddings.
4. **Hybrid Fusion** — weighted sum: `score = α * bm25_norm + (1-α) * dense_norm`. α=0.3 (dense-weighted default).

## Implementation (cleanrl style)

Single `train.py` with argparse, sectioned config, `log()` function, CSV metrics, multi-panel PNG figures. Output to `/content/rag-fasttext-output/`.

## Metrics

NDCG@10 (primary), MAP@100, Recall@100, MRR, per-query latency ms, index size MB.

## Visualization

4-panel figure: NDCG@10 bars by method, Recall@100 bars by method, Precision-Recall curve, latency histogram.

## Files

- `train.py` — full implementation
- `launch.py` — Colab bootstrap (pip install fasttext faiss-cpu scipy datasets matplotlib scikit-learn + spawn)
- `fetch.sh` — cron download script (3 min interval, excludes .pt checkpoints)

## Dependencies

`fasttext`, `faiss-cpu`, `scipy`, `numpy`, `datasets`, `matplotlib`, `scikit-learn`

## Training time (Colab CPU)

| Step | Time |
|------|------|
| Data load | ~5s |
| Tokenization | ~10s |
| FastText train | ~30-60s |
| BM25 index | ~2s |
| FAISS index | ~1s |
| Eval (323 queries) | ~10s |
| **Total** | **~1-2 min** |
