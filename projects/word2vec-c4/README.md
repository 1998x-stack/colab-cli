# Word2Vec-C4

Skip-gram with Negative Sampling (Word2Vec) trained on the C4 (Colossal Clean Crawled Corpus) English dataset, implemented from scratch in PyTorch.

## Usage

```bash
# Local training (medium preset: 300d, 50K vocab, 5 epochs, 2M sentences)
python train.py

# Override with size preset
python train.py --size small    # 100d, 20K vocab, 3 epochs, 500K sentences
python train.py --size large    # 300d, 100K vocab, 10 epochs, 5M sentences

# Or override individual settings
python train.py --embed-dim 200 --epochs 3 --max-sentences 500000
```

This project does not include a `launch.py` for Colab deployment.

## Key results

No completed training results in this repository. The training pipeline was interrupted during vocabulary building (streaming 2M sentences from C4).

Estimated setup for the medium preset:

| Setting | Value |
|---------|-------|
| Embedding dimension | 300 |
| Vocabulary size | 50,000 |
| Window size | 5 (dynamic) |
| Negative samples | 5 |
| Epochs | 5 |
| Max sentences | 2,000,000 |
| Subsampling threshold | 1e-5 |
| Min count | 5 |
| Batch size | 1024 |
| Optimizer | SGD with linear LR decay (0.003 to 0.00003) |

## Gotchas

- Pure PyTorch implementation -- no gensim, no external word2vec library. The model uses separate input (`in_emb`) and output (`out_emb`) embedding matrices.
- Dynamic window sampling: the actual window size is randomly sampled from [1, cfg.window] for each target word, following Mikolov et al.
- Subsampling uses the Mikolov formula: `p_keep = (sqrt(f/t) + 1) * (t/f)` with discard threshold 1e-5, which aggressively downsamples very frequent words.
- Noise distribution for negative sampling uses unigram frequency raised to the 3/4 power (PAD and UNK excluded).
- C4 is streamed from HuggingFace in two passes: first to build vocab (count frequencies), second to tokenize with subsampling. This avoids loading the full dataset into memory.
- Only alphabetic words (regex `[a-zA-Z]+`) are kept -- numbers, punctuation, and non-English characters are filtered out.
- LR follows a linear decay from `cfg.lr` to `cfg.lr * lr_end_ratio` (default 0.01) over the estimated total number of batches.
- The medium preset on CPU would take many hours; a GPU is strongly recommended for any training that completes more than vocab building.
- Checkpoints save only embeddings (no optimizer state), suitable for downstream evaluation but not for exact resume.
