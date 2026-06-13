"""Train a char-level GPT on tiny Shakespeare.

Outputs per CLAUDE.md spec:
  logs/train.log       -- per-epoch, timestamped, self-contained
  pngs/training_curves.png -- loss + perplexity over time
  metrics.csv          -- epoch, loss, perplexity, tokens_per_sec, elapsed_s
  checkpoints/         -- weights-only checkpoints

Usage:
  python train.py                          # defaults (CPU, 4-layer)
  python train.py --device cuda --n_layer 6  # GPU, larger model
"""
import csv
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import TransformerConfig, parse_args
from model import GPT

# --- Shakespeare download ---
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare(data_dir: str) -> str:
    """Download and load tiny Shakespeare text."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        print("[data] Downloading tiny Shakespeare (~1MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        return f.read()


class CharDataset(Dataset):
    """Character-level dataset: sliding windows over the text."""

    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def make_dirs(output_dir: str):
    for sub in ["logs", "pngs", "checkpoints"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, kv_caches=None)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def save_weights(path: str, model: nn.Module, epoch: int, metrics: dict):
    torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def main():
    config = parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Config: n_layer={config.n_layer}, d_model={config.d_model}, max_epochs={config.max_epochs}")

    make_dirs(config.output_dir)

    # --- Data ---
    text = load_shakespeare("/tmp/shakespeare")
    dataset = CharDataset(text, config.block_size)
    config.vocab_size = dataset.vocab_size

    n_train = int(0.9 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print(f"[data] Vocab size: {dataset.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --- Model ---
    model = GPT(config, use_cache=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Params: {n_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    total_steps = config.max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Logging setup ---
    log_path = os.path.join(config.output_dir, "logs", "train.log")
    csv_path = os.path.join(config.output_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "perplexity", "tokens_per_sec", "elapsed_s", "lr"])

    def log(msg: str):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # --- Training ---
    start_time = time.time()
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        tokens_processed = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits, _ = model(x, kv_caches=None)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            tokens_processed += x.numel()
            global_step += 1

            if global_step % config.log_interval == 0:
                log(
                    f"Ep {epoch}/{config.max_epochs} | "
                    f"step {global_step:5d} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # End of epoch
        epoch_elapsed = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        total_elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / epoch_elapsed
        perplexity = math.exp(min(avg_train_loss, 10))

        log(
            f"EPOCH {epoch}/{config.max_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ppl={perplexity:.1f} | "
            f"tok/s={tokens_per_sec:.0f} | "
            f"elapsed={total_elapsed/60:.1f}m"
        )

        # Write metrics
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_train_loss, val_loss, perplexity,
                tokens_per_sec, total_elapsed, scheduler.get_last_lr()[0],
            ])

        # Save weights checkpoint
        ckpt_path = os.path.join(config.output_dir, "checkpoints", f"weights_epoch{epoch}.pt")
        save_weights(ckpt_path, model, epoch, {
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "perplexity": perplexity,
        })
        log(f"Checkpoint saved: {ckpt_path}")

        # Generate training curves chart
        if epoch % config.chart_interval == 0:
            _make_training_chart(config.output_dir)

    total_elapsed = time.time() - start_time
    log(f"Training complete. Total: {total_elapsed/60:.1f}m. Final val loss: {val_loss:.4f}")


def _make_training_chart(output_dir: str):
    """Generate training_curves.png from metrics.csv (headless-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "b-", label="Train", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[1]
    ax.plot(df["epoch"], df["perplexity"], "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity")
    ax.grid(True, alpha=0.3)

    # Tokens/sec
    ax = axes[2]
    ax.plot(df["epoch"], df["tokens_per_sec"], "purple", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Speed")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_dir = os.path.join(output_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    fig.savefig(os.path.join(png_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
