"""Merge 12 experiment CSVs and generate comparison artifacts.

Usage: python analyze.py [--input-dir output/merged]

Reads output/merged/bs*_lr*/metrics.csv from all experiments,
merges into all_experiments.csv, generates 4 comparison plots.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_experiments(input_dir):
    """Scan input_dir for bs*_lr*/metrics.csv and load all."""
    experiments = []
    for exp_dir in sorted(Path(input_dir).glob("bs*_lr*")):
        csv_path = exp_dir / "metrics.csv"
        summary_path = exp_dir / "summary.json"
        if not csv_path.exists():
            continue

        name = exp_dir.name
        bs_part, lr_part = name.split("_lr")
        bs = int(bs_part.replace("bs", ""))
        lr = float(lr_part)

        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        experiments.append({
            "bs": bs, "lr": lr, "name": name,
            "rows": rows, "summary": summary,
        })

    return sorted(experiments, key=lambda e: (e["bs"], e["lr"]))


def write_comparison_csv(experiments, out_path):
    """Merge all experiments into one comparison CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bs", "lr", "best_test_acc", "final_test_acc", "final_test_loss",
            "steps_completed", "total_time_s", "grad_norm_std", "diverged",
        ])
        writer.writeheader()

        for exp in experiments:
            rows = exp["rows"]
            if not rows:
                continue

            best_acc = max(float(r["test_acc"]) for r in rows if r["test_acc"])
            final = rows[-1]
            grad_norms = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and float(r["grad_norm"]) > 0]
            grad_std = np.std(grad_norms) if grad_norms else 0

            writer.writerow({
                "bs": exp["bs"],
                "lr": exp["lr"],
                "best_test_acc": round(best_acc, 4),
                "final_test_acc": round(float(final.get("test_acc", 0)), 4),
                "final_test_loss": round(float(final.get("test_loss", 0)), 4),
                "steps_completed": exp["summary"].get("steps_completed", len(rows) * 200),
                "total_time_s": exp["summary"].get("total_time_s", 0),
                "grad_norm_std": round(grad_std, 4),
                "diverged": 1 if exp["summary"].get("steps_completed", 4000) < 100 else 0,
            })
    print(f"[analyze] Wrote {out_path}")


def plot_heatmap(experiments, out_path):
    """LR x BS -> best test accuracy heatmap."""
    bs_vals = sorted(set(e["bs"] for e in experiments))
    lr_vals = sorted(set(e["lr"] for e in experiments))

    data = np.zeros((len(bs_vals), len(lr_vals)))
    annot = []
    for i, bs in enumerate(bs_vals):
        row_annot = []
        for j, lr in enumerate(lr_vals):
            match = [e for e in experiments if e["bs"] == bs and e["lr"] == lr]
            if match and match[0]["rows"]:
                acc = max(float(r["test_acc"]) for r in match[0]["rows"] if r["test_acc"])
                data[i, j] = acc
                row_annot.append(f"{acc:.3f}")
            else:
                data[i, j] = np.nan
                row_annot.append("?")
        annot.append(row_annot)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.1, vmax=0.8)
    ax.set_xticks(range(len(lr_vals)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lr_vals])
    ax.set_yticks(range(len(bs_vals)))
    ax.set_yticklabels([f"BS={bs}" for bs in bs_vals])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Batch Size")
    ax.set_title("Best Test Accuracy: LR x BS Heatmap")

    for i in range(len(bs_vals)):
        for j in range(len(lr_vals)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11, color=color)

    plt.colorbar(im, ax=ax, label="Test Accuracy")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Heatmap -> {out_path}")


def plot_overlay_curves(experiments, out_path):
    """3 panels (one per BS), each with 4 LR curves overlaid."""
    bs_vals = sorted(set(e["bs"] for e in experiments))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LR Effect per Batch Size - CIFAR-10 SmallCNN", fontsize=13, fontweight="bold")

    for ax_idx, bs in enumerate(bs_vals):
        ax = axes[ax_idx]
        bs_exps = [e for e in experiments if e["bs"] == bs]

        for exp_idx, exp in enumerate(bs_exps):
            rows = exp["rows"]
            if not rows:
                continue
            xs = [int(r["batch"]) for r in rows if r["test_acc"]]
            ys = [float(r["test_acc"]) for r in rows if r["test_acc"]]
            if xs and ys:
                lr_idx = list(sorted(set(e["lr"] for e in experiments))).index(exp["lr"])
                ax.plot(xs, ys, "o-", color=colors[lr_idx], linewidth=1.2, markersize=2,
                       label=f"LR={exp['lr']:.0e}")

        ax.set_xlabel("Batch")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"BS={bs}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Overlay curves -> {out_path}")


def plot_optimal_lr_vs_bs(experiments, out_path):
    """Scatter: best LR per batch size with linear scaling reference."""
    bs_vals = sorted(set(e["bs"] for e in experiments))
    best_lrs = []

    for bs in bs_vals:
        bs_exps = [e for e in experiments if e["bs"] == bs]
        best_acc = -1
        best_lr = None
        for exp in bs_exps:
            rows = exp["rows"]
            if not rows:
                continue
            acc = max(float(r["test_acc"]) for r in rows if r["test_acc"])
            if acc > best_acc:
                best_acc = acc
                best_lr = exp["lr"]
        if best_lr:
            best_lrs.append((bs, best_lr, best_acc))

    if not best_lrs:
        print("[analyze] No data for optimal LR vs BS plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bss, lrs, accs = zip(*best_lrs)
    ax.scatter(bss, lrs, s=100, c=accs, cmap="RdYlGn", edgecolors="black", zorder=5)

    if len(best_lrs) >= 2:
        ref_bs = best_lrs[0][0]
        ref_lr = best_lrs[0][1]
        bs_range = np.linspace(min(bss) * 0.5, max(bss) * 1.5, 100)
        ax.plot(bs_range, [ref_lr * (b / ref_bs) for b in bs_range],
                "k--", alpha=0.5, linewidth=1, label="Linear scaling (LR proportional to BS)")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Optimal Learning Rate")
    ax.set_title("Optimal LR vs Batch Size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for bs, lr, acc in best_lrs:
        ax.annotate(f"acc={acc:.3f}", (bs, lr), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Optimal LR vs BS -> {out_path}")


def plot_gradient_noise(experiments, out_path):
    """Gradient norm std vs batch size."""
    bs_vals = sorted(set(e["bs"] for e in experiments))

    fig, ax = plt.subplots(figsize=(8, 5))

    for bs in bs_vals:
        bs_exps = [e for e in experiments if e["bs"] == bs]
        lrs = []
        grads = []
        for exp in bs_exps:
            rows = exp["rows"]
            if not rows:
                continue
            norms = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and float(r["grad_norm"]) > 0]
            if norms:
                lrs.append(exp["lr"])
                grads.append(np.std(norms))
        if lrs and grads:
            ax.scatter([bs] * len(lrs), grads, s=60, alpha=0.7)
            for lr, gstd in zip(lrs, grads):
                ax.annotate(f"LR={lr:.0e}", (bs, gstd), fontsize=7, alpha=0.8,
                           textcoords="offset points", xytext=(8, 0))

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Gradient Norm StdDev")
    ax.set_title("Gradient Noise vs Batch Size (lower BS = noisier gradients)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Gradient noise -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/merged",
                       help="Directory containing bs*_lr*/ subdirectories")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[analyze] Input dir {input_path} not found")
        sys.exit(1)

    experiments = load_all_experiments(input_path)
    print(f"[analyze] Loaded {len(experiments)} experiments")

    if not experiments:
        print("[analyze] No experiments found - nothing to analyze")
        sys.exit(1)

    os.makedirs("output/comparison", exist_ok=True)

    write_comparison_csv(experiments, "output/comparison/all_experiments.csv")
    plot_heatmap(experiments, "output/comparison/lr_bs_heatmap.png")
    plot_overlay_curves(experiments, "output/comparison/lr_curves.png")
    plot_optimal_lr_vs_bs(experiments, "output/comparison/optimal_lr_vs_bs.png")
    plot_gradient_noise(experiments, "output/comparison/gradient_noise.png")

    print("\n[analyze] All artifacts written to output/comparison/")


if __name__ == "__main__":
    main()
