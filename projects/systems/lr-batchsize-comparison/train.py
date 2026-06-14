"""LR × Batch Size experiment — single run.

Usage: python train.py --bs 16 --lr 1e-3

Runs 4000 optimizer steps with constant LR, evaluates every 200 steps.
Writes logs/train.log, metrics.csv, pngs/loss_acc.png, summary.json
to /content/lr-bs-output/bs<BS>_lr<LR>/
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as D


def setup_dirs(out_dir):
    for sub in ["logs", "pngs"]:
        Path(out_dir, sub).mkdir(parents=True, exist_ok=True)


def get_data(batch_size):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    test_ds = D.CIFAR10(root="/content/data", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()

    lr_str = f"{args.lr:.0e}".replace("e-0", "e-")
    out_dir = f"/content/lr-bs-output/bs{args.bs}_lr{lr_str}"
    setup_dirs(out_dir)

    log_path = f"{out_dir}/logs/train.log"
    csv_path = f"{out_dir}/metrics.csv"
    png_path = f"{out_dir}/pngs/loss_acc.png"

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            log_fh.write(line + "\n")

        log_msg(f"LR×BS experiment: bs={args.bs} lr={args.lr}")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch {torch.__version__}")

        torch.manual_seed(42)
        train_loader, test_loader = get_data(args.bs)

        model = SmallCNN().cuda()
        init_loss = None

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=[
                "batch", "loss", "train_acc", "test_loss", "test_acc", "lr", "grad_norm", "elapsed_s"
            ])
            csv_w.writeheader()

            t0 = time.time()
            batch_losses = []
            eval_points = []

            train_iter = iter(train_loader)
            for batch_idx in range(1, 4001):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.cuda(), y.cuda()

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = F.cross_entropy(out, y)

                if init_loss is None:
                    init_loss = loss.item()
                    log_msg(f"Initial loss: {init_loss:.4f}  (expected ~2.30 for CIFAR-10)")

                # Divergence check: loss > 3× initial → LR too high (Karpathy heuristic)
                if loss.item() > init_loss * 3:
                    log_msg(f"DIVERGED at batch {batch_idx}: loss={loss.item():.4f} > 3×init={init_loss*3:.4f}")
                    csv_w.writerow({
                        "batch": batch_idx, "loss": round(loss.item(), 6),
                        "train_acc": 0.0, "test_loss": 0.0, "test_acc": 0.0,
                        "lr": args.lr, "grad_norm": 0.0, "elapsed_s": round(time.time() - t0, 1),
                    })
                    break

                scaler.scale(loss).backward()

                # Log unclipped gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                train_acc = (out.argmax(1) == y).float().mean().item()
                batch_losses.append(loss.item())

                # Eval every 200 batches
                if batch_idx % 200 == 0:
                    test_loss, test_acc = evaluate(model, test_loader, "cuda")
                    elapsed = time.time() - t0
                    log_msg(
                        f"Batch {batch_idx:>5} | loss={loss.item():.4f} | "
                        f"train_acc={train_acc:.3f} | test_loss={test_loss:.4f} | "
                        f"test_acc={test_acc:.3f} | grad_norm={total_norm:.2f} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    csv_w.writerow({
                        "batch": batch_idx,
                        "loss": round(loss.item(), 6),
                        "train_acc": round(train_acc, 4),
                        "test_loss": round(test_loss, 4),
                        "test_acc": round(test_acc, 4),
                        "lr": args.lr,
                        "grad_norm": round(total_norm, 4),
                        "elapsed_s": round(elapsed, 1),
                    })
                    eval_points.append((batch_idx, test_acc))
                    model.train()

                    # Generate plot every 1000 batches
                    if batch_idx % 1000 == 0:
                        try:
                            _save_plot(batch_losses, eval_points, png_path, args)
                        except Exception:
                            pass

        total_time = time.time() - t0
        final_test_loss, final_test_acc = evaluate(model, test_loader, "cuda")
        best_acc = max((a for _, a in eval_points), default=final_test_acc)

        log_msg(f"DONE: final_test_acc={final_test_acc:.4f} best_acc={best_acc:.4f} time={total_time:.0f}s")

        # Write summary
        summary = {
            "bs": args.bs, "lr": args.lr, "steps_completed": batch_idx,
            "init_loss": round(init_loss, 4) if init_loss else None,
            "final_test_acc": round(final_test_acc, 4),
            "best_acc": round(best_acc, 4),
            "total_time_s": round(total_time, 1),
        }
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Final plot
        try:
            _save_plot(batch_losses, eval_points, png_path, args)
        except Exception:
            pass


def _save_plot(batch_losses, eval_points, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    suptitle = f"BS={args.bs}  LR={args.lr}  —  CIFAR-10 SmallCNN"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    # Loss
    ax = axes[0]
    if batch_losses:
        w = min(50, len(batch_losses))
        if len(batch_losses) >= w:
            smooth = np.convolve(batch_losses, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(batch_losses)), smooth, color="darkorange", linewidth=1.2, label=f"avg{w}")
        ax.plot(batch_losses, alpha=0.12, color="steelblue", linewidth=0.3)
    ax.set_xlabel("Batch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if eval_points:
        xs, ys = zip(*eval_points)
        ax.plot(xs, ys, "o-", color="mediumseagreen", linewidth=1.5, markersize=3)
        best = max(ys)
        ax.axhline(y=best, color="green", linestyle=":", alpha=0.7, label=f"Best={best:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Batch"); ax.set_ylabel("Test Accuracy"); ax.set_title("Evaluation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
