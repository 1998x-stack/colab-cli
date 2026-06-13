"""Training loop for Seq2Seq LSTM on Multi30k EN→DE — Sutskever et al. 2014.

High-signal outputs (CLAUDE.md §3): logs/train.log, pngs/training_curves.png, metrics.csv

Usage:
    python train.py
    python train.py --epochs 20 --batch_size 64 --reverse_src
    python train.py --resume /content/checkpoints/ckpt_epoch5.pt
"""
import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import build_seq2seq, PAD, SOS, EOS
from dataset import load_multi30k, build_tokenizer, build_dataloaders

# --- Paths ---
BASE_DIR = "/content/seq2seq-t4"
LOG_DIR = os.path.join(BASE_DIR, "logs")
PNGS_DIR = os.path.join(BASE_DIR, "pngs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
METRICS_CSV = os.path.join(BASE_DIR, "metrics.csv")
LOG_PATH = os.path.join(LOG_DIR, "train.log")
CURVES_PNG = os.path.join(PNGS_DIR, "training_curves.png")


def setup_dirs():
    for d in [LOG_DIR, PNGS_DIR, CKPT_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


class TeeLogger:
    """Write to both stdout and log file."""
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.stdout.write(msg)
        self.log.write(msg)

    def flush(self):
        self.stdout.flush()
        self.log.flush()


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def save_curves(history: dict, path: str):
    """Multi-panel training dashboard."""
    try:
        _save_curves_impl(history, path)
    except Exception as e:
        log(f"WARNING: save_curves failed: {e}")


def _save_curves_impl(history: dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    steps = history["steps"]
    losses = history["losses"]
    ppls = history["ppls"]
    grad_norms = history["grad_norms"]
    bleu_steps = history.get("bleu_steps", [])
    bleus = history.get("bleus", [])

    # Loss
    ax = axes[0, 0]
    ax.plot(steps, losses, "b-", alpha=0.7, linewidth=1.0)
    # Moving average
    if len(losses) >= 50:
        ma = np.convolve(losses, np.ones(50)/50, mode="valid")
        max_plot = min(len(steps), len(ma))
        ax.plot(steps[-max_plot:], ma[-max_plot:], "r-", linewidth=1.5, label="MA(50)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[0, 1]
    ax.plot(steps, ppls, "g-", alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity (e^loss)")
    ax.set_title("Perplexity")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 0]
    ax.plot(steps, grad_norms, "orange", alpha=0.7, linewidth=1.0)
    ax.axhline(y=5.0, color="r", linestyle="--", alpha=0.5, label="clip=5.0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # BLEU
    ax = axes[1, 1]
    if bleu_steps:
        ax.plot(bleu_steps, bleus, "m-o", markersize=4, linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("BLEU")
    ax.set_title("Val BLEU (greedy)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path: str, history: dict):
    """Write full history to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "ppl", "grad_norm", "val_bleu", "elapsed_s"])

        bleu_by_step = dict(zip(history.get("bleu_steps", []), history.get("bleus", [])))

        for i, step in enumerate(history["steps"]):
            bleu = bleu_by_step.get(step, "")
            writer.writerow([
                step,
                round(history["losses"][i], 4),
                round(history["ppls"][i], 4),
                round(history["grad_norms"][i], 4),
                bleu,
                round(history["elapsed"][i], 1),
            ])


@torch.no_grad()
def evaluate_bleu(model, val_loader, tgt_tokenizer, device, max_samples: int = 200):
    """Greedy decode on val set, compute BLEU with sacrebleu."""
    try:
        import sacrebleu
    except ImportError:
        log("sacrebleu not installed — skipping BLEU eval")
        return 0.0

    model.eval()
    refs, hyps = [], []
    count = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        outputs = model.greedy_decode(src)

        for i in range(src.size(0)):
            if count >= max_samples:
                break
            # Reference: tgt without SOS/EOS, decode to text
            ref_tokens = [t for t in tgt[i].tolist() if t not in (PAD, SOS, EOS)]
            ref_text = tgt_tokenizer.decode(ref_tokens)
            # Hypothesis: decoded tokens
            hyp_tokens = [t for t in outputs[i].tolist() if t not in (PAD, SOS, EOS)]
            hyp_text = tgt_tokenizer.decode(hyp_tokens)
            if ref_text.strip() and hyp_text.strip():
                refs.append(ref_text)
                hyps.append(hyp_text)
            count += 1
        if count >= max_samples:
            break

    if not hyps:
        return 0.0

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a")
    model.train()
    return bleu.score


def train(args):
    setup_dirs()
    sys.stdout = TeeLogger(LOG_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Args: {vars(args)}")

    # --- Data ---
    pairs = load_multi30k(DATA_DIR, reverse_src=args.reverse_src)

    src_tokenizer = build_tokenizer(pairs["train"], args.vocab_size, lang="src")
    tgt_tokenizer = build_tokenizer(pairs["train"], args.vocab_size, lang="tgt")

    loaders = build_dataloaders(pairs, src_tokenizer, tgt_tokenizer,
                                batch_size=args.batch_size, src_max_len=args.max_len,
                                tgt_max_len=args.max_len)

    src_vocab = src_tokenizer.get_vocab_size()
    tgt_vocab = tgt_tokenizer.get_vocab_size()

    # --- Model ---
    model = build_seq2seq(src_vocab, tgt_vocab,
                          embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, dropout=args.dropout,
                          device=device)

    # Paper: no padding index masking needed — PAD token gets zero loss via ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # Paper: SGD lr=0.7, halved every epoch after epoch 5 (but our scale is different)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    start_epoch = 0
    total_steps = 0

    # --- Resume ---
    if args.resume:
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        total_steps = ckpt.get("step", 0)
        log(f"Resumed at epoch {start_epoch}, step {total_steps}")

    # --- History ---
    history = {"steps": [], "losses": [], "ppls": [], "grad_norms": [],
               "bleu_steps": [], "bleus": [], "elapsed": []}
    t_start = time.time()

    # --- Training ---
    log(f"Training {args.epochs} epochs, {len(loaders['train'])} batches/epoch")
    log(f"  batch_size={args.batch_size}, lr={args.lr}, reverse_src={args.reverse_src}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()

        for batch_idx, (src, tgt) in enumerate(loaders["train"]):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Teacher forcing: feed all but last token, predict all but SOS
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, tgt_vocab), tgt[:, 1:].reshape(-1))
            loss.backward()

            # Paper: gradient clipping at 5.0
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            total_steps += 1
            epoch_loss += loss.item()

            # Log every N steps
            if total_steps % args.log_every == 0 or (epoch == 0 and batch_idx == 0):
                ppl = np.exp(loss.item())
                elapsed = time.time() - t_start
                log(f"Ep {epoch+1}/{args.epochs} | step {total_steps} | "
                    f"loss={loss.item():.3f} | ppl={ppl:.1f} | "
                    f"grad_norm={grad_norm:.2f} | elapsed={elapsed:.0f}s")

                history["steps"].append(total_steps)
                history["losses"].append(loss.item())
                history["ppls"].append(ppl)
                history["grad_norms"].append(grad_norm.item())
                history["elapsed"].append(elapsed)

                # Save curves every N steps
                if total_steps % (args.log_every * 5) == 0:
                    save_curves(history, CURVES_PNG)

        # --- End of epoch ---
        avg_loss = epoch_loss / len(loaders["train"])
        ppl = np.exp(avg_loss)
        elapsed = time.time() - t_epoch
        log(f"=== Epoch {epoch+1} done | avg_loss={avg_loss:.3f} | "
            f"ppl={ppl:.1f} | epoch_time={elapsed:.0f}s ===")

        # BLEU eval
        val_bleu = evaluate_bleu(model, loaders["val"], tgt_tokenizer, device)
        log(f"=== Val BLEU (greedy): {val_bleu:.1f} ===")

        history["bleu_steps"].append(total_steps)
        history["bleus"].append(val_bleu)
        save_curves(history, CURVES_PNG)

        scheduler.step()

        # Checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "src_vocab_size": src_vocab,
            "tgt_vocab_size": tgt_vocab,
        }, ckpt_path)
        log(f"Saved checkpoint: {ckpt_path}")

        # Weights-only checkpoint (smaller — for download through proxy)
        weights_path = os.path.join(CKPT_DIR, f"weights_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), weights_path)

    # --- Final ---
    total_time = time.time() - t_start
    log(f"Training complete. Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    log(f"Final BLEU: {history['bleus'][-1]:.1f}" if history["bleus"] else "No BLEU")

    write_metrics_csv(METRICS_CSV, history)
    save_curves(history, CURVES_PNG)
    log(f"Metrics → {METRICS_CSV}")
    log(f"Curves → {CURVES_PNG}")


def main():
    parser = argparse.ArgumentParser(description="Seq2Seq LSTM — Multi30k EN→DE")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.7)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--reverse_src", action="store_true", default=True,
                        help="Reverse source word order (paper's key insight)")
    parser.add_argument("--no_reverse_src", action="store_false", dest="reverse_src")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
