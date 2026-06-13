"""T4-adapted data prep: TinyStories dataset + BPE tokenizer.

Downloads karpathy/tinystories-gpt4-clean from HuggingFace,
trains a small BPE tokenizer (vocab=2048), saves to ~/.cache/autoresearch-t4/.
One-time run. ~2-3 min on Colab T4.
"""

import os
import time
import argparse
import pickle

import torch
from datasets import load_dataset
import rustbpe
import tiktoken

# ── T4-friendly constants ────────────────────────────────────────────────────
MAX_SEQ_LEN = 256
TIME_BUDGET = 300
EVAL_TOKENS = 4 * 524288
VOCAB_SIZE = 2048

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-t4")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def download_data(max_chars=200_000_000):
    """Download TinyStories and save as text shards."""
    os.makedirs(DATA_DIR, exist_ok=True)
    shard_path = os.path.join(DATA_DIR, "tinystories.txt")
    if os.path.exists(shard_path):
        print(f"Data: already downloaded at {shard_path}")
        return

    print("Data: downloading TinyStories from HuggingFace...")
    ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
    with open(shard_path, "w") as f:
        total = 0
        for ex in ds:
            text = ex["text"].strip()
            if len(text) < 50:
                continue
            f.write(text + "\n\n")
            total += len(text)
            if total >= max_chars:
                break
    size_mb = os.path.getsize(shard_path) / 1024 / 1024
    print(f"Data: saved {total:,} chars to {shard_path} ({size_mb:.1f} MB)")


def text_iterator(max_chars=100_000_000):
    """Yield stories from the downloaded data."""
    shard_path = os.path.join(DATA_DIR, "tinystories.txt")
    nchars = 0
    with open(shard_path) as f:
        for line in f:
            line = line.strip()
            if line:
                nchars += len(line)
                yield line
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe", pat_str=pattern,
        mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: vocab_size={enc.n_vocab}")


# ── Runtime utilities (imported by train.py) ──────────────────────────────────

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            return ids
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    return torch.load(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), map_location=device)


def make_dataloader(tokenizer, B, T, split="train", buffer_size=500):
    """BOS-aligned dataloader with best-fit packing. Every row starts with BOS."""
    assert split in ["train", "val"]
    shard = os.path.join(DATA_DIR, "tinystories.txt")
    with open(shard) as f:
        docs = [line.strip() for line in f if line.strip()]

    split_idx = int(0.9 * len(docs))
    docs = docs[:split_idx] if split == "train" else docs[split_idx:]

    bos_id = tokenizer.get_bos_token_id()
    row_capacity = T + 1
    epoch = 1

    # Pre-allocate GPU buffers
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    # Row buffer: T+1 wide so we can shift 1 for targets
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    doc_idx = 0
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                remaining = row_capacity - pos
                best_doc = None
                best_len = 0
                attempts = 0
                while best_doc is None and attempts < min(50, len(docs)):
                    doc_text = docs[doc_idx % len(docs)]
                    doc_idx += 1
                    tokens = tokenizer.encode(doc_text, prepend=bos_id)
                    if len(tokens) <= remaining and len(tokens) > best_len:
                        best_doc = tokens
                        best_len = len(tokens)
                    attempts += 1

                if best_doc is not None:
                    row_buffer[row_idx, pos:pos + len(best_doc)] = torch.tensor(best_doc, dtype=torch.long)
                    pos += len(best_doc)
                else:
                    # No doc fits: crop shortest to fill exactly
                    doc_text = docs[doc_idx % len(docs)]
                    doc_idx += 1
                    tokens = tokenizer.encode(doc_text, prepend=bos_id)[:remaining]
                    row_buffer[row_idx, pos:pos + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                    pos += len(tokens)

        cpu_inputs.copy_(row_buffer[:, :T])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch
        epoch += 1


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


import math  # for log(2) in evaluate_bpb

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    download_data(max_chars=args.max_chars)
    print()
    train_tokenizer()
    print()
    print("Done! Ready to train.")
