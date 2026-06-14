#!/usr/bin/env python3
"""Post-hoc comparison visualization: Post-LN vs Pre-LN training curves.

Reads metrics.csv from both output directories and generates a 4-panel
comparison figure.

Usage:
    python charts.py
    python charts.py --post-ln output/postln --pre-ln output/preln --out comparison.png
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(csv_path: str) -> dict[str, list]:
    """Read metrics.csv into column lists."""
    if not os.path.exists(csv_path):
        return {}

    cols = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in cols:
                    cols[key] = []
                try:
                    cols[key].append(float(val))
                except (ValueError, TypeError):
                    cols[key].append(val)
    return cols


def moving_average(data: list[float], window: int = 50) -> list[float]:
    if len(data) < window:
        return data
    return list(np.convolve(data, np.ones(window) / window, mode="valid"))


def plot_comparison(post: dict, pre: dict, out_path: str, window: int = 50):
    """4-panel figure comparing Post-LN vs Pre-LN training."""
    if not post or not pre:
        print("Missing data — need both post-LN and pre-LN metrics.csv")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    post_steps = [int(s) for s in post.get("step", [])]
    pre_steps = [int(s) for s in pre.get("step", [])]
    post_loss = post.get("loss", [])
    pre_loss = pre.get("loss", [])
    post_lr = post.get("lr", [])
    pre_lr = pre.get("lr", [])
    post_val = post.get("val_loss", [])
    pre_val = pre.get("val_loss", [])

    # --- Panel 1: Overlaid training loss ---
    ax = axes[0, 0]
    if post_steps and post_loss:
        ax.plot(post_steps, post_loss, alpha=0.2, color="#2563eb", linewidth=0.5)
        if len(post_loss) >= window:
            ma = moving_average(post_loss, window)
            ax.plot(post_steps[window-1:], ma, color="#2563eb", linewidth=2,
                    label=f"Post-LN (MA-{window})")
    if pre_steps and pre_loss:
        ax.plot(pre_steps, pre_loss, alpha=0.2, color="#ea580c", linewidth=0.5)
        if len(pre_loss) >= window:
            ma = moving_average(pre_loss, window)
            ax.plot(pre_steps[window-1:], ma, color="#ea580c", linewidth=2,
                    label=f"Pre-LN (MA-{window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss — Post-LN vs Pre-LN")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Validation loss ---
    ax = axes[0, 1]
    if post_steps and post_val:
        ax.plot(post_steps, post_val, color="#2563eb", linewidth=1.5,
                marker=".", markersize=3, label="Post-LN")
    if pre_steps and pre_val:
        ax.plot(pre_steps, pre_val, color="#ea580c", linewidth=1.5,
                marker=".", markersize=3, label="Pre-LN")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Learning rate ---
    ax = axes[1, 0]
    if post_steps and post_lr:
        ax.plot(post_steps, post_lr, color="#2563eb", linewidth=1, label="Post-LN")
    if pre_steps and pre_lr:
        ax.plot(pre_steps, pre_lr, color="#ea580c", linewidth=1, label="Pre-LN")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Bar comparison (filter NaN) ---
    ax = axes[1, 1]
    labels = []
    values = []
    colors = []
    if post_loss and not math.isnan(post_loss[-1]):
        labels.append("Post-LN\n(train)")
        values.append(post_loss[-1])
        colors.append("#2563eb")
    if pre_loss and not math.isnan(pre_loss[-1]):
        labels.append("Pre-LN\n(train)")
        values.append(pre_loss[-1])
        colors.append("#ea580c")
    if post_val and not math.isnan(post_val[-1]):
        labels.append("Post-LN\n(val)")
        values.append(post_val[-1])
        colors.append("#60a5fa")
    if pre_val and not math.isnan(pre_val[-1]):
        labels.append("Pre-LN\n(val)")
        values.append(pre_val[-1])
        colors.append("#fdba74")

    if values:
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Final Loss Comparison")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Post-LN vs Pre-LN Transformer — IWSLT2017 DE-EN",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison saved: {out_path}")


def print_summary(post: dict, pre: dict):
    """Print a text summary table."""
    def last_or_na(d, key):
        vals = d.get(key, [])
        return f"{vals[-1]:.4f}" if vals else "N/A"

    print()
    print("=" * 60)
    print("  Post-LN vs Pre-LN — Summary")
    print("=" * 60)
    print(f"  {'':20s} {'Post-LN':>15s} {'Pre-LN':>15s}")
    print(f"  {'-'*50}")
    print(f"  {'Steps':20s} {int(post.get('step', [[0]])[-1] or 0):>15d} {int(pre.get('step', [[0]])[-1] or 0):>15d}")
    print(f"  {'Final Train Loss':20s} {last_or_na(post, 'loss'):>15s} {last_or_na(pre, 'loss'):>15s}")
    print(f"  {'Final Val Loss':20s} {last_or_na(post, 'val_loss'):>15s} {last_or_na(pre, 'val_loss'):>15s}")
    print(f"  {'Final LR':20s} {last_or_na(post, 'lr'):>15s} {last_or_na(pre, 'lr'):>15s}")

    # Winner determination
    post_final = post.get("loss", [])
    pre_final = pre.get("loss", [])
    if post_final and pre_final:
        post_val = post_final[-1]
        pre_val = pre_final[-1]
        if isinstance(post_val, float) and math.isnan(post_val):
            print(f"\n  >>> Pre-LN converges stably (loss={pre_val:.4f}) — Post-LN diverged to NaN")
        elif isinstance(pre_val, float) and math.isnan(pre_val):
            print(f"\n  >>> Post-LN converges stably (loss={post_val:.4f}) — Pre-LN diverged to NaN")
        elif pre_val < post_val:
            print(f"\n  >>> Pre-LN achieves lower training loss ({pre_val:.4f} vs {post_val:.4f})")
        else:
            print(f"\n  >>> Post-LN achieves lower training loss ({post_val:.4f} vs {pre_val:.4f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Post-LN vs Pre-LN comparison charts"
    )
    parser.add_argument("--post-ln", default="output/postln",
                        help="Post-LN output directory")
    parser.add_argument("--pre-ln", default="output/preln",
                        help="Pre-LN output directory")
    parser.add_argument("--out", default="output/comparison.png",
                        help="Output figure path")
    parser.add_argument("--window", type=int, default=50,
                        help="Moving average window")
    args = parser.parse_args()

    post = read_metrics(os.path.join(args.post_ln, "metrics.csv"))
    pre = read_metrics(os.path.join(args.pre_ln, "metrics.csv"))

    if not post:
        print(f"WARNING: No post-LN metrics found at {args.post_ln}/metrics.csv")
    if not pre:
        print(f"WARNING: No pre-LN metrics found at {args.pre_ln}/metrics.csv")

    print_summary(post, pre)
    plot_comparison(post, pre, args.out, args.window)


if __name__ == "__main__":
    main()
