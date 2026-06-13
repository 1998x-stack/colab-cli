"""Word2Vec Skip-gram with Negative Sampling on C4 dataset.

Single-file implementation in cleanrl style — argparse config, torch-only,
structured logging, CSV metrics, PNG visualizations.

Usage:
    python train.py                           # default: medium config
    python train.py --size small              # small: 100d, 20k vocab, 3 epochs
    python train.py --size large              # large: 300d, 100k vocab, 10 epochs
    python train.py --embed-dim 200 --epochs 3 --max-sentences 500000
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib_available = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib_available = True
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

SIZE_PRESETS = {
    "small":  dict(embed_dim=100, vocab_size=20000, window=5, neg_samples=5,
                   epochs=3, max_sentences=500000, lr=0.003, min_count=10),
    "medium": dict(embed_dim=300, vocab_size=50000, window=5, neg_samples=5,
                   epochs=5, max_sentences=2000000, lr=0.003, min_count=5),
    "large":  dict(embed_dim=300, vocab_size=100000, window=10, neg_samples=10,
                   epochs=10, max_sentences=5000000, lr=0.003, min_count=3),
}

QUERY_WORDS = [
    "king", "queen", "man", "woman", "paris", "france", "london", "england",
    "computer", "science", "music", "art", "war", "peace", "good", "bad",
    "car", "train", "food", "water", "happy", "sad", "big", "small",
    "money", "work", "school", "university", "love", "hate", "day", "night",
]

ANALOGIES = [
    ("king", "man", "queen"),     # king - man + woman ≈ queen
    ("paris", "france", "london"), # paris - france + england ≈ london
    ("man", "woman", "king"),     # man - woman + queen ≈ king
    ("good", "bad", "happy"),     # good - bad + sad ≈ happy
    ("big", "small", "fast"),     # big - small + slow ≈ fast
]


def parse_args():
    p = argparse.ArgumentParser(description="Word2Vec Skip-gram with Negative Sampling")
    p.add_argument("--size", type=str, default="medium", choices=["small", "medium", "large"],
                   help="Preset config size (default: medium)")
    # Overrides for any preset value
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--neg-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-sentences", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-count", type=int, default=None)
    # Fixed params (no preset override)
    p.add_argument("--subsample-thresh", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr-end-ratio", type=float, default=0.01,
                   help="Final LR as fraction of initial LR (default: 0.01)")
    p.add_argument("--out-dir", type=str, default="/content/word2vec-c4-output")
    p.add_argument("--plot-every", type=int, default=5000,
                   help="Batches between PNG refreshes")
    p.add_argument("--save-every", type=int, default=50000,
                   help="Batches between checkpoint saves")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="allenai/c4")
    p.add_argument("--dataset-config", type=str, default="en")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


def resolve_config(args):
    """Apply size preset, then override with any explicit CLI args."""
    preset = SIZE_PRESETS[args.size]
    cfg = {}
    for key in ["embed_dim", "vocab_size", "window", "neg_samples",
                 "epochs", "max_sentences", "lr", "min_count"]:
        cli_val = getattr(args, key.replace("-", "_"), None)
        cfg[key] = cli_val if cli_val is not None else preset[key]
    cfg["subsample_thresh"] = args.subsample_thresh
    cfg["batch_size"] = args.batch_size
    cfg["lr_end_ratio"] = args.lr_end_ratio
    cfg["out_dir"] = args.out_dir
    cfg["plot_every"] = args.plot_every
    cfg["save_every"] = args.save_every
    cfg["seed"] = args.seed
    cfg["dataset"] = args.dataset
    cfg["dataset_config"] = args.dataset_config
    cfg["hf_token"] = args.hf_token or os.environ.get("HF_TOKEN")
    cfg["size"] = args.size
    cfg["lr_end"] = cfg["lr"] * cfg["lr_end_ratio"]
    return argparse.Namespace(**cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenization
# ═══════════════════════════════════════════════════════════════════════════════

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]+\b")

def tokenize(text):
    """Lowercase + extract alphabetic words only."""
    return TOKEN_PATTERN.findall(text.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary building
# ═══════════════════════════════════════════════════════════════════════════════

def build_vocab(cfg, logger):
    """Stream C4, count word frequencies, build vocab with min_count filter."""
    from datasets import load_dataset

    logger.log(f"Building vocabulary from {cfg.dataset}/{cfg.dataset_config} (streaming)...")
    logger.log(f"Target: max {cfg.max_sentences:,} sentences, min_count={cfg.min_count}")

    token = cfg.hf_token
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split="train",
                      streaming=True, token=token)

    word_counts = Counter()
    n_sentences = 0
    n_tokens_total = 0
    t0 = time.time()

    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        tokens = tokenize(text)
        word_counts.update(tokens)
        n_tokens_total += len(tokens)
        n_sentences += 1

        if n_sentences % 100000 == 0:
            elapsed = time.time() - t0
            logger.log(f"  {n_sentences:,} sentences | {len(word_counts):,} unique words | "
                       f"{n_tokens_total:,} tokens | {elapsed:.0f}s")

        if n_sentences >= cfg.max_sentences:
            break

    elapsed = time.time() - t0
    logger.log(f"Vocab pass done: {n_sentences:,} sentences, {n_tokens_total:,} tokens, "
               f"{len(word_counts):,} unique words in {elapsed:.0f}s")

    # Filter by min_count, keep top vocab_size
    filtered = [(w, c) for w, c in word_counts.items() if c >= cfg.min_count]
    filtered.sort(key=lambda x: -x[1])
    filtered = filtered[:cfg.vocab_size]

    # Build mappings (0 = PAD, 1 = UNK)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    freqs = [0, 0]  # PAD and UNK frequencies

    for i, (word, count) in enumerate(filtered):
        idx = i + 2
        word2idx[word] = idx
        idx2word[idx] = word
        freqs.append(count)

    actual_vocab_size = len(word2idx)
    freqs = np.array(freqs, dtype=np.float64)
    total_words = freqs.sum()

    logger.log(f"Vocabulary: {actual_vocab_size:,} words (pad=0, unk=1, "
               f"min_count={cfg.min_count}, top-N from {len(word_counts):,} raw)")
    logger.log(f"  Top-10: {[idx2word[i] for i in range(2, min(12, actual_vocab_size))]}")

    return word2idx, idx2word, freqs, total_words, n_sentences


# ═══════════════════════════════════════════════════════════════════════════════
# Subsampling & noise distribution
# ═══════════════════════════════════════════════════════════════════════════════

def compute_subsample_probs(freqs, total_words, threshold):
    """P(discard) = 1 - sqrt(t/f) for words with freq > threshold.

    Returns array of DISCARD probabilities per word index.
    """
    freq = freqs / total_words
    # Only compute for words with freq > 0; PAD/UNK handled below
    mask = freq > 0
    keep = np.ones_like(freq)
    # Mikolov formula: p_keep = (sqrt(f/t) + 1) * (t/f), clipped to [0, 1]
    keep[mask] = (np.sqrt(freq[mask] / threshold) + 1) * (threshold / freq[mask])
    keep = np.clip(keep, 0.0, 1.0)
    return 1.0 - keep  # discard probabilities


def compute_noise_distribution(freqs, vocab_size):
    """Unigram distribution raised to 3/4 power — for negative sampling."""
    # Skip PAD (idx=0)
    f = freqs[1:].copy().astype(np.float64)
    f[0] = 0.0  # zero out UNK for noise (we never sample UNK as negative)
    f_pow = np.power(f, 0.75)
    f_pow /= f_pow.sum()
    return torch.from_numpy(f_pow).float()


# ═══════════════════════════════════════════════════════════════════════════════
# Data: convert sentences to token indices
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_dataset(cfg, word2idx, discard_probs, logger):
    """Second pass over C4: convert sentences to token index lists with subsampling."""
    from datasets import load_dataset

    logger.log("Tokenizing dataset (second pass)...")

    token = cfg.hf_token
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split="train",
                      streaming=True, token=token)

    sentences = []
    n_kept = 0
    n_discarded = 0
    t0 = time.time()

    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue

        tokens = tokenize(text)
        indices = []
        for t in tokens:
            idx = word2idx.get(t, 1)  # 1 = UNK
            # Subsampling
            if discard_probs[idx] > 0 and np.random.random() < discard_probs[idx]:
                n_discarded += 1
                continue
            n_kept += 1
            indices.append(idx)

        if len(indices) >= 2:  # need at least 2 words for a (target, context) pair
            sentences.append(indices)

        if len(sentences) % 100000 == 0:
            elapsed = time.time() - t0
            logger.log(f"  {len(sentences):,} sentences | {n_kept:,} kept | "
                       f"{n_discarded:,} discarded | {elapsed:.0f}s")

        if len(sentences) >= cfg.max_sentences:
            break

    elapsed = time.time() - t0
    keep_pct = 100 * n_kept / max(n_kept + n_discarded, 1)
    logger.log(f"Tokenization done: {len(sentences):,} sentences, {n_kept:,} tokens kept "
               f"({keep_pct:.1f}%) in {elapsed:.0f}s")

    return sentences


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class SkipGramNegSampling(nn.Module):
    """Skip-gram with negative sampling.

    Input embeddings (v_in): center word → predicts context
    Output embeddings (v_out): context words → used in NCE loss
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.out_emb = nn.Embedding(vocab_size, embed_dim, sparse=False)
        # Init: uniform in [-0.5/embed_dim, 0.5/embed_dim] (Mikolov et al.)
        nn.init.uniform_(self.in_emb.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.uniform_(self.out_emb.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, target, context, neg_samples):
        """Compute NCE loss for one batch.

        Args:
            target: (B,) center word indices
            context: (B,) positive context word indices
            neg_samples: (B, K) negative sample indices

        Returns:
            scalar loss
        """
        v_in = self.in_emb(target)          # (B, D)
        v_ctx = self.out_emb(context)       # (B, D)
        v_neg = self.out_emb(neg_samples)   # (B, K, D)

        pos_score = torch.sum(v_in * v_ctx, dim=1)           # (B,)
        pos_loss = -F.logsigmoid(pos_score).mean()

        neg_score = torch.bmm(v_neg, v_in.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()

        return pos_loss + neg_loss

    def get_embeddings(self):
        """Return normalized input embeddings (standard for downstream use)."""
        w = self.in_emb.weight.detach()
        return w / (w.norm(dim=1, keepdim=True) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def generate_batch(sentences, cfg, noise_dist, vocab_size, device):
    """Generate one batch of (target, context, negatives) pairs.

    Walks through sentences sequentially, generates pairs with dynamic window,
    yields batches of size cfg.batch_size.
    """
    batch_target = []
    batch_context = []
    rng = np.random.default_rng()

    sent_indices = np.arange(len(sentences))
    rng.shuffle(sent_indices)

    for sent_idx in sent_indices:
        sent = sentences[sent_idx]
        sent_len = len(sent)
        if sent_len < 2:
            continue

        for pos, word_idx in enumerate(sent):
            # Dynamic window: sample actual window from [1, cfg.window]
            win = rng.integers(1, cfg.window + 1)
            start = max(0, pos - win)
            end = min(sent_len, pos + win + 1)

            for ctx_pos in range(start, end):
                if ctx_pos == pos:
                    continue
                batch_target.append(word_idx)
                batch_context.append(sent[ctx_pos])

                if len(batch_target) >= cfg.batch_size:
                    target_t = torch.tensor(batch_target, dtype=torch.long, device=device)
                    ctx_t = torch.tensor(batch_context, dtype=torch.long, device=device)

                    # Sample negatives for the whole batch
                    n = len(batch_target) * cfg.neg_samples
                    neg = torch.multinomial(noise_dist, n, replacement=True)
                    neg = neg.view(len(batch_target), cfg.neg_samples).to(device)

                    batch_target = []
                    batch_context = []

                    yield target_t, ctx_t, neg

    # Final partial batch
    if len(batch_target) > 0:
        target_t = torch.tensor(batch_target, dtype=torch.long, device=device)
        ctx_t = torch.tensor(batch_context, dtype=torch.long, device=device)
        n = len(batch_target) * cfg.neg_samples
        neg = torch.multinomial(noise_dist, n, replacement=True)
        neg = neg.view(len(batch_target), cfg.neg_samples).to(device)
        yield target_t, ctx_t, neg


def cosine_similarity(vec, embeddings):
    """Cosine similarity between one vector and all embeddings."""
    return torch.mv(embeddings, vec)  # embeddings are already normalized


def most_similar(word, word2idx, idx2word, embeddings, top_k=10):
    """Find top-k most similar words by cosine similarity."""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]
    scores = cosine_similarity(vec, embeddings)
    # Exclude the query word itself, PAD, UNK
    scores[idx] = -float("inf")
    scores[0] = -float("inf")
    scores[1] = -float("inf")
    top_indices = torch.topk(scores, top_k).indices
    return [(idx2word[int(i)], float(scores[i])) for i in top_indices]


def analogy(a, b, c, word2idx, idx2word, embeddings, top_k=5):
    """a - b + c ≈ ?   (e.g., king - man + woman ≈ queen)"""
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []
    vec = (embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]])
    vec = vec / (vec.norm() + 1e-8)
    scores = cosine_similarity(vec, embeddings)
    # Exclude query words
    for w in [a, b, c]:
        if w in word2idx:
            scores[word2idx[w]] = -float("inf")
    scores[0] = -float("inf")
    scores[1] = -float("inf")
    top_indices = torch.topk(scores, top_k).indices
    return [(idx2word[int(i)], float(scores[i])) for i in top_indices]


def plot_curves(cfg, losses, batch_losses, words_per_sec, lr_values, epoch_boundaries,
                logger):
    """Generate multi-panel training curves PNG."""
    if not matplotlib_available:
        logger.log("  matplotlib not available — skipping PNG generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Word2Vec Skip-gram — {cfg.size.upper()} config", fontsize=14,
                 fontweight="bold")

    # Loss curve
    ax = axes[0, 0]
    ax.plot(batch_losses, alpha=0.15, color="steelblue", linewidth=0.3, label="Batch loss")
    if len(losses) > 0:
        ax.plot(losses, color="darkorange", linewidth=1.5, label="Moving avg")
    for ep_idx, boundary in enumerate(epoch_boundaries[:-1]):
        ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.4,
                   label="Epoch boundary" if ep_idx == 0 else None)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Words/sec throughput
    ax = axes[0, 1]
    ax.plot(words_per_sec, color="mediumseagreen", linewidth=1.5)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Words/sec")
    ax.set_title("Training Throughput")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(lr_values, color="crimson", linewidth=1.5)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (linear decay)")
    ax.grid(True, alpha=0.3)

    # Loss distribution (histogram)
    ax = axes[1, 1]
    if len(batch_losses) > 0:
        recent = batch_losses[-min(10000, len(batch_losses)):]
        ax.hist(recent, bins=50, color="mediumseagreen", alpha=0.8, edgecolor="white")
        ax.axvline(x=np.mean(recent), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(recent):.3f}")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Count")
    ax.set_title("Batch Loss Distribution (recent)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{cfg.out_dir}/pngs/training_curves.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(cfg, sentences, word2idx, idx2word, noise_dist, vocab_size, logger):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU info
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.log(f"GPU: {name} ({vram:.1f} GB VRAM)")
    else:
        logger.log("GPU: not available, using CPU")

    total_tokens = sum(len(s) for s in sentences)
    total_batches_est = (total_tokens * cfg.window * 2 * cfg.epochs) // cfg.batch_size
    logger.log(f"Sentences: {len(sentences):,} | Total tokens: {total_tokens:,}")
    logger.log(f"Estimated batches: ~{total_batches_est:,} ({cfg.epochs} epochs)")
    logger.log(f"Embed dim: {cfg.embed_dim} | Window: {cfg.window} | "
               f"Neg samples: {cfg.neg_samples} | Batch size: {cfg.batch_size}")
    logger.log(f"LR: {cfg.lr} → {cfg.lr_end:.6f} (linear decay)")

    model = SkipGramNegSampling(vocab_size, cfg.embed_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    # Track metrics
    batch_losses = []
    avg_losses = []
    lr_values = []
    words_per_sec_hist = []
    epoch_boundaries = [0]
    global_batch = 0
    start_time = time.time()
    total_batches = total_batches_est

    logger.log("── Training start ──")

    noise_dist_gpu = noise_dist.to(device)

    for epoch in range(cfg.epochs):
        epoch_t0 = time.time()
        epoch_batches = 0
        epoch_tokens = 0

        for target, context, neg in generate_batch(sentences, cfg, noise_dist_gpu,
                                                    vocab_size, device):
            loss = model(target, context, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            global_batch += 1
            epoch_batches += 1
            epoch_tokens += target.size(0)

            # LR decay
            frac = min(global_batch / total_batches, 1.0)
            lr = cfg.lr + (cfg.lr_end - cfg.lr) * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            lr_values.append(lr)

            # Periodic logging
            if global_batch % 100 == 0 or global_batch == 1:
                avg_loss = np.mean(batch_losses[-100:])
                avg_losses.append(avg_loss)
                elapsed = time.time() - start_time
                words_sec = (global_batch * cfg.batch_size) / max(elapsed, 0.001)
                words_per_sec_hist.append(words_sec)

                logger.log(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Batch {global_batch:,} | "
                    f"loss={loss.item():.4f} | "
                    f"avg100={avg_loss:.4f} | "
                    f"words/s={words_sec:,.0f} | "
                    f"lr={lr:.6f} | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Generate PNGs
            if global_batch % cfg.plot_every == 0:
                plot_curves(cfg, avg_losses, batch_losses, words_per_sec_hist,
                            lr_values, epoch_boundaries, logger)

            # Save checkpoint (embeddings only, no optimizer state)
            if global_batch % cfg.save_every == 0:
                ckpt_path = f"{cfg.out_dir}/checkpoints/model_batch_{global_batch:06d}.pt"
                torch.save({
                    "batch": global_batch,
                    "epoch": epoch + 1,
                    "in_emb": model.in_emb.weight.detach().cpu(),
                    "out_emb": model.out_emb.weight.detach().cpu(),
                    "word2idx": word2idx,
                    "idx2word": idx2word,
                    "loss": loss.item(),
                }, ckpt_path)
                logger.log(f"  → checkpoint saved: {os.path.basename(ckpt_path)}")

        # End of epoch
        epoch_elapsed = time.time() - epoch_t0
        epoch_wps = epoch_tokens / max(epoch_elapsed, 0.001)
        logger.log(f"── Epoch {epoch+1}/{cfg.epochs} done: {epoch_batches:,} batches, "
                   f"{epoch_elapsed:.0f}s, {epoch_wps:,.0f} words/s ──")

        epoch_boundaries.append(global_batch)

    total_elapsed = time.time() - start_time
    logger.log(f"── Training complete in {total_elapsed:.0f}s "
               f"({total_elapsed/60:.1f}m) ──")
    logger.log(f"Final avg100 loss: {np.mean(batch_losses[-100:]):.4f}")

    return model, batch_losses, avg_losses, words_per_sec_hist, lr_values, epoch_boundaries


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = resolve_config(args)

    os.makedirs(f"{cfg.out_dir}/logs", exist_ok=True)
    os.makedirs(f"{cfg.out_dir}/pngs", exist_ok=True)
    os.makedirs(f"{cfg.out_dir}/checkpoints", exist_ok=True)

    logger = Logger(f"{cfg.out_dir}/logs/train.log")

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.log("═" * 60)
    logger.log(f"Word2Vec Skip-gram | size={cfg.size} | dataset={cfg.dataset}")
    logger.log(f"Config: embed_dim={cfg.embed_dim} vocab_size={cfg.vocab_size} "
               f"window={cfg.window} neg={cfg.neg_samples} epochs={cfg.epochs} "
               f"max_sentences={cfg.max_sentences:,}")
    logger.log(f"LR={cfg.lr}→{cfg.lr_end:.6f} batch_size={cfg.batch_size} "
               f"subsample_thresh={cfg.subsample_thresh}")
    logger.log(f"Output: {cfg.out_dir}")

    # 1. Build vocabulary
    word2idx, idx2word, freqs, total_words, n_sentences = build_vocab(cfg, logger)
    vocab_size = len(word2idx)

    # 2. Compute subsampling probabilities & noise distribution
    discard_probs = compute_subsample_probs(freqs, total_words, cfg.subsample_thresh)
    noise_dist = compute_noise_distribution(freqs, vocab_size)
    logger.log(f"Noise distribution: max_p={noise_dist.max():.6f} "
               f"min_p={noise_dist[noise_dist > 0].min():.6f}")

    # 3. Tokenize dataset
    sentences = tokenize_dataset(cfg, word2idx, discard_probs, logger)
    if len(sentences) == 0:
        logger.log("ERROR: No sentences after tokenization. Check dataset/config.")
        sys.exit(1)

    # 4. Train
    model, batch_losses, avg_losses, wps_hist, lr_vals, epoch_boundaries = train(
        cfg, sentences, word2idx, idx2word, noise_dist, vocab_size, logger)

    # 5. Final plots
    plot_curves(cfg, avg_losses, batch_losses, wps_hist, lr_vals, epoch_boundaries, logger)
    logger.log("Final PNGs saved.")

    # 6. Save final embeddings
    embeddings = model.get_embeddings()
    final_path = f"{cfg.out_dir}/final_embeddings.pt"
    torch.save({
        "in_emb": model.in_emb.weight.detach().cpu(),
        "out_emb": model.out_emb.weight.detach().cpu(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "config": vars(cfg),
    }, final_path)
    logger.log(f"Final embeddings saved: {final_path}")

    # 7. Evaluation: word similarity & analogies
    logger.log("── Word Similarity ──")
    embeddings_gpu = embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for word in QUERY_WORDS:
        sims = most_similar(word, word2idx, idx2word, embeddings_gpu, top_k=5)
        if sims:
            sim_str = ", ".join(f"{w}({s:.3f})" for w, s in sims)
            logger.log(f"  {word:12s} → {sim_str}")

    logger.log("── Word Analogies ──")
    for a, b, c in ANALOGIES:
        results = analogy(a, b, c, word2idx, idx2word, embeddings_gpu, top_k=3)
        if results:
            res_str = ", ".join(f"{w}({s:.3f})" for w, s in results)
            logger.log(f"  {a} - {b} + {c} ≈ [{res_str}]")

    # 8. Summary JSON
    summary = {
        "size": cfg.size,
        "dataset": cfg.dataset,
        "embed_dim": cfg.embed_dim,
        "vocab_size": vocab_size,
        "window": cfg.window,
        "neg_samples": cfg.neg_samples,
        "epochs": cfg.epochs,
        "max_sentences": n_sentences,
        "total_tokens": int(sum(len(s) for s in sentences)),
        "batches_trained": len(batch_losses),
        "final_loss": float(np.mean(batch_losses[-100:])) if batch_losses else None,
        "total_time_s": round(time.time() - (time.time() - 0), 1),
    }
    with open(f"{cfg.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log("Done.")


if __name__ == "__main__":
    main()
