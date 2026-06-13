"""FastText classifier in PyTorch — subword n-gram embeddings + linear classifier.

Paper: "Enriching Word Vectors with Subword Information" (Bojanowski et al., 2017)
Architecture: word → word_emb + mean(ngram_embs) → mean across doc → linear → class

Single-file cleanrl style: argparse config, torch-only model, no trainer abstractions.

Usage:
    python train.py                           # default: small config, full AG News
    python train.py --size tiny               # quick test: 30k docs, 50d, 3 epochs
    python train.py --embed-dim 200 --epochs 10
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

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
    "tiny": dict(
        embed_dim=50, vocab_size=20000, min_n=3, max_n=6, ngram_buckets=500_000,
        epochs=3, batch_size=128, lr=0.1, min_count=10, max_docs=30000,
        max_ngrams_per_word=15,
    ),
    "small": dict(
        embed_dim=100, vocab_size=50000, min_n=3, max_n=6, ngram_buckets=2_000_000,
        epochs=5, batch_size=64, lr=0.1, min_count=5, max_docs=120000,
        max_ngrams_per_word=20,
    ),
    "cpu": dict(
        embed_dim=50, vocab_size=30000, min_n=3, max_n=6, ngram_buckets=200_000,
        epochs=5, batch_size=64, lr=0.1, min_count=5, max_docs=120000,
        max_ngrams_per_word=15,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FastText classifier — subword n-gram embeddings + linear classifier"
    )
    p.add_argument("--size", type=str, default="cpu", choices=["tiny", "small", "cpu"],
                   help="Preset config (default: cpu — optimized for Colab CPU)")
    # Overrides
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--min-n", type=int, default=None)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--ngram-buckets", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-count", type=int, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-ngrams-per-word", type=int, default=None)
    # Fixed params
    p.add_argument("--lr-decay-ratio", type=float, default=0.01,
                   help="Final LR as fraction of initial (default: 0.01)")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dataset", type=str, default="fancyzhx/ag_news")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--out-dir", type=str, default="/content/fasttext-pytorch-output")
    p.add_argument("--log-every", type=int, default=100,
                   help="Batches between log lines")
    p.add_argument("--plot-every", type=int, default=500,
                   help="Batches between PNG refreshes")
    p.add_argument("--eval-every", type=int, default=1000,
                   help="Batches between eval runs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_config(args):
    preset = SIZE_PRESETS[args.size]
    cfg = {}
    for key in ["embed_dim", "vocab_size", "min_n", "max_n", "ngram_buckets",
                 "epochs", "batch_size", "lr", "min_count", "max_docs",
                 "max_ngrams_per_word"]:
        cli_val = getattr(args, key.replace("-", "_"), None)
        cfg[key] = cli_val if cli_val is not None else preset[key]
    cfg["lr_decay_ratio"] = args.lr_decay_ratio
    cfg["weight_decay"] = args.weight_decay
    cfg["dataset"] = args.dataset
    cfg["out_dir"] = args.out_dir
    cfg["log_every"] = args.log_every
    cfg["plot_every"] = args.plot_every
    cfg["eval_every"] = args.eval_every
    cfg["seed"] = args.seed
    cfg["size"] = args.size
    cfg["lr_end"] = cfg["lr"] * cfg["lr_decay_ratio"]

    # HF token: CLI arg > env var > ~/.huggingface/token file
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        for p in [os.path.expanduser("~/.huggingface/token"),
                  os.path.expanduser("~/.huggingface/access_token")]:
            if os.path.exists(p):
                with open(p) as f:
                    hf_token = f.read().strip()
                break
    cfg["hf_token"] = hf_token

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
# Tokenization & N-grams
# ═══════════════════════════════════════════════════════════════════════════════

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]+\b")


def tokenize(text):
    """Lowercase + extract alphabetic words."""
    return TOKEN_PATTERN.findall(text.lower())


def _fnv_hash(s):
    """FNV-1a 64-bit hash — deterministic across runs."""
    h = 14695981039346656037
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def compute_word_ngrams(word, min_n, max_n, ngram_buckets, max_ngrams):
    """Compute n-gram bucket indices for a word with boundary markers.

    Word "hello" with min_n=3, max_n=6 → n-grams of "<he", "hel", "ell", ... "llo>", "ello>"
    Each n-gram is hashed to a bucket [0, ngram_buckets).
    """
    w = f"<{word}>"
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(w) < n:
            break
        for i in range(len(w) - n + 1):
            ng = w[i:i + n]
            bucket = _fnv_hash(ng) % ngram_buckets + 1  # +1: reserve 0 for PAD
            ngrams.append(bucket)

    if len(ngrams) == 0:
        return [1]  # fallback: single UNK-like n-gram

    # Truncate to max_ngrams (deterministic: keep first N)
    return ngrams[:max_ngrams]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading & Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_data(cfg, logger):
    """Load AG News from HuggingFace datasets. Returns (train_texts, train_labels,
    test_texts, test_labels)."""
    from datasets import load_dataset

    logger.log(f"Loading dataset: {cfg.dataset} ...")
    t0 = time.time()

    ds = load_dataset(cfg.dataset, token=cfg.hf_token)
    # ag_news has 'train' and 'test' splits

    train_texts = []
    train_labels = []
    for row in ds["train"]:
        text = row.get("text", "") or ""
        label = row.get("label", 0)
        train_texts.append(text)
        train_labels.append(label)
        if len(train_texts) >= cfg.max_docs:
            break

    test_texts = []
    test_labels = []
    for row in ds["test"]:
        text = row.get("text", "") or ""
        label = row.get("label", 0)
        test_texts.append(text)
        test_labels.append(label)

    elapsed = time.time() - t0
    logger.log(f"Loaded {len(train_texts):,} train + {len(test_texts):,} test docs in {elapsed:.0f}s")
    logger.log(f"Classes: {len(set(train_labels))}")

    return train_texts, train_labels, test_texts, test_labels


def build_vocab(train_texts, cfg, logger):
    """Build vocabulary from tokenized training texts.

    Returns:
        word2idx: dict word → index (0=PAD, 1=UNK, 2..N=real words)
        idx2word: dict index → word
        word_ngrams: dict word_idx → list of n-gram bucket indices
        ngram_table_size: actual max ngram bucket index used + 1
    """
    logger.log("Tokenizing training texts for vocabulary...")
    t0 = time.time()

    # Count words
    word_counts = Counter()
    n_tokens = 0
    for text in train_texts:
        tokens = tokenize(text)
        word_counts.update(tokens)
        n_tokens += len(tokens)

    elapsed = time.time() - t0
    logger.log(f"  {n_tokens:,} tokens, {len(word_counts):,} unique words in {elapsed:.0f}s")

    # Filter and build vocabulary
    filtered = [(w, c) for w, c in word_counts.items() if c >= cfg.min_count]
    filtered.sort(key=lambda x: -x[1])
    filtered = filtered[:cfg.vocab_size]

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for i, (word, _) in enumerate(filtered):
        idx = i + 2
        word2idx[word] = idx
        idx2word[idx] = word

    vocab_size = len(word2idx)
    logger.log(f"Vocabulary: {vocab_size:,} words (min_count={cfg.min_count}, "
               f"top-{cfg.vocab_size} from {len(word_counts):,} raw)")

    # Precompute n-gram hashes for each vocabulary word
    logger.log("Precomputing n-gram hashes for vocabulary...")
    t0 = time.time()
    word_ngrams = {}
    max_bucket = 0
    for word, idx in word2idx.items():
        if idx < 2:  # skip PAD and UNK
            word_ngrams[idx] = []
            continue
        ngs = compute_word_ngrams(word, cfg.min_n, cfg.max_n, cfg.ngram_buckets, cfg.max_ngrams_per_word)
        word_ngrams[idx] = ngs
        if ngs:
            max_bucket = max(max_bucket, max(ngs))

    ngram_table_size = max_bucket + 2  # +1 for safety, +1 for 0-index
    elapsed = time.time() - t0
    avg_ngrams = np.mean([len(v) for v in word_ngrams.values()])
    logger.log(f"  N-grams precomputed: avg {avg_ngrams:.1f}/word, "
               f"max_bucket={max_bucket:,}, table_size={ngram_table_size:,} "
               f"in {elapsed:.0f}s")

    # Show top words for sanity
    top_words = [idx2word[i] for i in range(2, min(12, vocab_size))]
    logger.log(f"  Top-10: {top_words}")

    return word2idx, idx2word, word_ngrams, ngram_table_size


def prepare_dataset(texts, labels, word2idx, word_ngrams, cfg):
    """Convert texts to (word_ids, ngram_ids, label) tensors.

    Returns list of tuples: (word_ids_tensor, ngram_ids_tensor, label_int)
    """
    data = []

    for text, label in zip(texts, labels):
        tokens = tokenize(text)
        if len(tokens) == 0:
            continue

        word_ids = []
        ngram_ids = []
        max_ng = cfg.max_ngrams_per_word

        for token in tokens:
            wid = word2idx.get(token, 1)  # 1 = UNK
            word_ids.append(wid)

            ngs = word_ngrams.get(wid, word_ngrams.get(1, []))
            # Pad to max_ngrams_per_word with 0
            padded = ngs[:max_ng] + [0] * max(0, max_ng - len(ngs))
            ngram_ids.append(padded[:max_ng])

        word_ids_t = torch.tensor(word_ids, dtype=torch.long)
        ngram_ids_t = torch.tensor(ngram_ids, dtype=torch.long)
        data.append((word_ids_t, ngram_ids_t, label))

    return data


def collate_batch(batch):
    """Pad and stack a batch of (word_ids, ngram_ids, label) tuples."""
    max_len = max(item[0].size(0) for item in batch)
    max_ng = batch[0][1].size(1)
    B = len(batch)

    word_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ng, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)

    for i, (wids, nids, lbl) in enumerate(batch):
        L = wids.size(0)
        word_ids[i, :L] = wids
        ngram_ids[i, :L] = nids
        labels[i] = lbl

    word_mask = (word_ids != 0)
    return word_ids, ngram_ids, word_mask, labels


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class FastTextClassifier(nn.Module):
    """FastText for text classification.

    Words → word_embedding + mean(character n-gram embeddings)
    Documents → mean(word vectors)
    Classification → Linear(doc_vector, num_classes)
    """

    def __init__(self, vocab_size, ngram_table_size, embed_dim, num_classes):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.ngram_emb = nn.Embedding(ngram_table_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

        # Init: uniform [-0.1, 0.1] (matching original fastText)
        nn.init.uniform_(self.word_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.ngram_emb.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Zero out padding
        self.word_emb.weight.data[0] = 0.0
        self.ngram_emb.weight.data[0] = 0.0

    def forward(self, word_ids, ngram_ids, word_mask):
        """
        Args:
            word_ids: (B, L) word indices, 0=PAD
            ngram_ids: (B, L, N) n-gram bucket indices per word, 0=PAD
            word_mask: (B, L) True for real words

        Returns:
            logits: (B, num_classes)
        """
        B, L = word_ids.shape
        D = self.word_emb.embedding_dim

        # Word embeddings: (B, L, D)
        w_emb = self.word_emb(word_ids)

        # N-gram embeddings: average over n-grams per word
        _, _, N = ngram_ids.shape
        ng_flat = ngram_ids.reshape(-1, N)           # (B*L, N)
        ng_emb = self.ngram_emb(ng_flat)             # (B*L, N, D)
        ng_mask = (ng_flat != 0).float().unsqueeze(-1)  # (B*L, N, 1)
        ng_sum = (ng_emb * ng_mask).sum(dim=1)        # (B*L, D)
        ng_count = ng_mask.sum(dim=1).clamp(min=1)    # (B*L, 1)
        ng_mean = (ng_sum / ng_count).reshape(B, L, D)  # (B, L, D)

        # Word representation = word embedding + n-gram mean
        word_repr = w_emb + ng_mean                     # (B, L, D)

        # Document representation = mean of word representations
        wm = word_mask.float().unsqueeze(-1)            # (B, L, 1)
        doc_vec = (word_repr * wm).sum(dim=1)           # (B, D)
        doc_len = wm.sum(dim=1).clamp(min=1)            # (B, 1)
        doc_vec = doc_vec / doc_len                     # (B, D)

        logits = self.fc(doc_vec)                       # (B, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, data, cfg, device):
    """Compute loss and accuracy on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    indices = np.arange(len(data))
    with torch.no_grad():
        for i in range(0, len(data), cfg.batch_size):
            batch_indices = indices[i:i + cfg.batch_size]
            batch = [data[j] for j in batch_indices]
            word_ids, ngram_ids, word_mask, labels = collate_batch(batch)
            word_ids = word_ids.to(device)
            ngram_ids = ngram_ids.to(device)
            word_mask = word_mask.to(device)
            labels = labels.to(device)

            logits = model(word_ids, ngram_ids, word_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(batch_indices)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(batch_indices)

    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


def train_epoch(model, data, optimizer, criterion, cfg, device, epoch, total_batches_est,
                global_batch, start_time, logger, metrics):
    """Train one epoch. Mutates metrics dict in place."""
    indices = np.arange(len(data))
    np.random.default_rng(cfg.seed + epoch).shuffle(indices)

    n_batches = len(data) // cfg.batch_size
    epoch_loss = 0.0

    for batch_idx in range(n_batches):
        start = batch_idx * cfg.batch_size
        batch_indices = indices[start:start + cfg.batch_size]
        batch = [data[i] for i in batch_indices]

        word_ids, ngram_ids, word_mask, labels = collate_batch(batch)
        word_ids = word_ids.to(device)
        ngram_ids = ngram_ids.to(device)
        word_mask = word_mask.to(device)
        labels = labels.to(device)

        logits = model(word_ids, ngram_ids, word_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # LR linear decay
        frac = min(global_batch / total_batches_est, 1.0)
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss_val = loss.item()
        epoch_loss += loss_val
        global_batch += 1

        # Record metrics
        metrics["batch_losses"].append(loss_val)
        metrics["lr_values"].append(lr)

        # Logging
        if global_batch % cfg.log_every == 0 or global_batch == 1:
            avg100 = np.mean(metrics["batch_losses"][-100:])
            elapsed = time.time() - start_time
            logger.log(
                f"Ep {epoch+1}/{cfg.epochs} | "
                f"Batch {global_batch:,} | "
                f"loss={loss_val:.4f} | "
                f"avg100={avg100:.4f} | "
                f"lr={lr:.6f} | "
                f"elapsed={elapsed:.0f}s"
            )

        # Periodic eval
        if global_batch % cfg.eval_every == 0:
            eval_loss, eval_acc = evaluate(model, metrics["test_data"], cfg, device)
            metrics["eval_losses"].append((global_batch, eval_loss))
            metrics["eval_accs"].append((global_batch, eval_acc))
            logger.log(
                f"  ── Eval @ batch {global_batch:,}: "
                f"test_loss={eval_loss:.4f} | test_acc={eval_acc:.4f}"
            )

        # Periodic plots
        if global_batch % cfg.plot_every == 0:
            plot_curves(cfg, metrics, logger)

    avg_epoch_loss = epoch_loss / max(n_batches, 1)
    return global_batch, avg_epoch_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curves(cfg, metrics, logger):
    """Generate multi-panel training curves PNG."""
    if not matplotlib_available:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"FastText Classifier — AG News ({cfg.size.upper()} config)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Loss curve
    ax = axes[0, 0]
    batch_losses = metrics["batch_losses"]
    ax.plot(batch_losses, alpha=0.15, color="steelblue", linewidth=0.3, label="Batch loss")
    if len(batch_losses) >= 100:
        window = min(100, len(batch_losses))
        smooth = np.convolve(batch_losses, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(batch_losses)), smooth, color="darkorange",
                linewidth=1.5, label=f"Moving avg ({window})")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Test accuracy over time
    ax = axes[0, 1]
    if metrics["eval_accs"]:
        evals_x, evals_y = zip(*metrics["eval_accs"])
        ax.plot(evals_x, evals_y, "o-", color="mediumseagreen", linewidth=1.5, markersize=3)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Batch")
        # Reference lines
        ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90% baseline")
        best_acc = max(evals_y)
        ax.axhline(y=best_acc, color="green", linestyle=":", alpha=0.7,
                   label=f"Best={best_acc:.4f}")
        ax.legend(fontsize=8)
    ax.set_title("Test Accuracy (periodic eval)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Learning rate schedule
    ax = axes[1, 0]
    ax.plot(metrics["lr_values"], color="crimson", linewidth=1.5)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (linear decay)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Loss distribution (recent batches)
    ax = axes[1, 1]
    bl = metrics["batch_losses"]
    if len(bl) > 0:
        recent = bl[-min(2000, len(bl)):]
        ax.hist(recent, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(x=np.mean(recent), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(recent):.3f}")
        ax.axvline(x=np.median(recent), color="orange", linestyle="--", linewidth=1.5,
                   label=f"median={np.median(recent):.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Count")
    ax.set_title("Batch Loss Distribution (recent 2000)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{cfg.out_dir}/pngs/training_curves.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    logger.log("═" * 60)
    logger.log(f"FastText Classifier (PyTorch) | size={cfg.size} | dataset={cfg.dataset}")
    logger.log(f"Config: embed_dim={cfg.embed_dim} vocab={cfg.vocab_size} "
               f"ngram_buckets={cfg.ngram_buckets} min_n={cfg.min_n} max_n={cfg.max_n} "
               f"max_ngrams_per_word={cfg.max_ngrams_per_word}")
    logger.log(f"Training: epochs={cfg.epochs} batch_size={cfg.batch_size} "
               f"lr={cfg.lr}→{cfg.lr_end:.6f} max_docs={cfg.max_docs}")
    logger.log(f"Output: {cfg.out_dir}")

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 1. Load data
    train_texts, train_labels, test_texts, test_labels = load_raw_data(cfg, logger)

    # 2. Build vocabulary + precompute n-grams
    word2idx, idx2word, word_ngrams, ngram_table_size = build_vocab(train_texts, cfg, logger)
    vocab_size = len(word2idx)
    num_classes = len(set(train_labels))
    logger.log(f"Classes: {num_classes} | Vocab size: {vocab_size} | "
               f"N-gram table size: {ngram_table_size:,}")

    # 3. Prepare datasets (convert texts to index tensors)
    logger.log("Preparing training dataset (tokenize + n-gram lookup)...")
    t0 = time.time()
    train_data = prepare_dataset(train_texts, train_labels, word2idx, word_ngrams, cfg)
    logger.log(f"  Train: {len(train_data):,} valid docs in {time.time() - t0:.0f}s")

    logger.log("Preparing test dataset...")
    t0 = time.time()
    test_data = prepare_dataset(test_texts, test_labels, word2idx, word_ngrams, cfg)
    logger.log(f"  Test: {len(test_data):,} valid docs in {time.time() - t0:.0f}s")

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)} "
                   f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        logger.log("Device: CPU")

    model = FastTextClassifier(vocab_size, ngram_table_size, cfg.embed_dim, num_classes)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model params: {n_params:,}")

    # 5. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    batches_per_epoch = len(train_data) // cfg.batch_size
    total_batches_est = batches_per_epoch * cfg.epochs
    logger.log(f"Batches/epoch: ~{batches_per_epoch:,} | Total: ~{total_batches_est:,} "
               f"({cfg.epochs} epochs)")

    # Metrics tracking
    metrics = {
        "batch_losses": [],
        "lr_values": [],
        "eval_losses": [],
        "eval_accs": [],
        "epoch_losses": [],
        "epoch_accs": [],
        "test_data": test_data,
    }

    # CSV header
    csv_path = f"{cfg.out_dir}/metrics.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,elapsed_s,lr\n")

    # 6. Training loop
    logger.log("── Training start ──")
    global_batch = 0
    start_time = time.time()

    for epoch in range(cfg.epochs):
        epoch_t0 = time.time()

        global_batch, avg_loss = train_epoch(
            model, train_data, optimizer, criterion, cfg, device,
            epoch, total_batches_est, global_batch, start_time, logger, metrics
        )

        epoch_elapsed = time.time() - epoch_t0

        # Epoch-end evaluation
        train_loss, train_acc = evaluate(model, train_data, cfg, device)
        test_loss, test_acc = evaluate(model, test_data, cfg, device)

        total_elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(
            f"── Epoch {epoch+1}/{cfg.epochs} done ── "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | "
            f"time={total_elapsed:.0f}s ({epoch_elapsed:.0f}s/ep)"
        )

        # Write CSV row
        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},"
                    f"{test_acc:.6f},{total_elapsed:.1f},{current_lr:.6f}\n")

        metrics["epoch_losses"].append(train_loss)
        metrics["epoch_accs"].append(test_acc)

        # Save checkpoint (overwrite — ngram table is large, keep only latest)
        ckpt_path = f"{cfg.out_dir}/checkpoints/model_latest.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "word2idx": word2idx,
            "word_ngrams": word_ngrams,
            "ngram_table_size": ngram_table_size,
            "num_classes": num_classes,
            "config": vars(cfg),
            "test_acc": test_acc,
        }, ckpt_path)
        logger.log(f"  → checkpoint: {os.path.basename(ckpt_path)}")

    # 7. Training complete
    total_elapsed = time.time() - start_time
    logger.log(f"── Training complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m) ──")

    # 8. Final evaluation
    _, final_train_acc = evaluate(model, train_data, cfg, device)
    _, final_test_acc = evaluate(model, test_data, cfg, device)
    logger.log(f"Final accuracy — train: {final_train_acc:.4f} | test: {final_test_acc:.4f}")

    # 9. Final plots
    plot_curves(cfg, metrics, logger)

    # 10. Summary JSON
    summary = {
        "size": cfg.size,
        "dataset": cfg.dataset,
        "embed_dim": cfg.embed_dim,
        "vocab_size": vocab_size,
        "ngram_buckets": cfg.ngram_buckets,
        "ngram_table_size": ngram_table_size,
        "num_classes": num_classes,
        "epochs_completed": cfg.epochs,
        "batches_trained": global_batch,
        "n_params": n_params,
        "train_acc": round(final_train_acc, 4),
        "test_acc": round(final_test_acc, 4),
        "total_time_s": round(total_elapsed, 1),
        "config": vars(cfg),
    }
    with open(f"{cfg.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log(f"Summary saved. Test accuracy: {final_test_acc:.4f}")
    logger.log("Done.")


if __name__ == "__main__":
    main()
