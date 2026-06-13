"""Post-training charts: read output directory and produce visualizations.

Usage:
  python charts.py --output_dir output
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    png_dir = os.path.join(args.output_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    if os.path.exists(csv_path):
        _plot_training_curves(csv_path, png_dir)

    print(f"Charts saved to {png_dir}/")


def _plot_training_curves(csv_path, png_dir):
    epochs, train_loss, val_loss, ppl, tok_s = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            ppl.append(float(row["perplexity"]))
            tok_s.append(float(row["tokens_per_sec"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(epochs, train_loss, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, "r-", label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Perplexity
    axes[0, 1].plot(epochs, ppl, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Perplexity")
    axes[0, 1].set_title("Perplexity"); axes[0, 1].grid(alpha=0.3)

    # Tokens/sec
    axes[1, 0].plot(epochs, tok_s, "purple", linewidth=2)
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Tokens/sec")
    axes[1, 0].set_title("Training Speed"); axes[1, 0].grid(alpha=0.3)

    # Train/Val ratio (overfitting monitor)
    ratio = [t / max(v, 0.01) for t, v in zip(train_loss, val_loss)]
    axes[1, 1].plot(epochs, ratio, "orange", linewidth=2)
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Train/Val Loss Ratio")
    axes[1, 1].set_title("Overfitting Monitor"); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved training_curves.png")


if __name__ == "__main__":
    main()
