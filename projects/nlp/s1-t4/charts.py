"""Visualization: training loss curve and test-time scaling curve.

Reads train_loss.jsonl and metrics.json to produce publication-quality PNGs.

Usage:
    python charts.py
    python charts.py --loss /path/to/train_loss.jsonl --metrics /path/to/metrics.json --output_dir /path/to/output
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_train_loss(loss_path: str, output_dir: str):
    """Plot training loss curve from JSONL file.

    Reads train_loss.jsonl (one JSON per line: {"step": N, "loss": M}),
    plots raw loss (thin blue) and smoothed loss (thick orange), and
    saves train_loss.png at 150 DPI.

    Args:
        loss_path: Path to train_loss.jsonl.
        output_dir: Directory to save train_loss.png.
    """
    if not os.path.exists(loss_path):
        print(f"[charts] train_loss.jsonl not found: {loss_path}")
        return

    steps, losses = [], []
    with open(loss_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                steps.append(record["step"])
                losses.append(record["loss"])
            except (json.JSONDecodeError, KeyError):
                continue

    if not steps:
        print("[charts] train_loss.jsonl is empty -- skipping train_loss plot")
        return

    steps = np.array(steps)
    losses = np.array(losses)

    # Smoothed loss: rolling window of ~5% of data points, min window=5
    window = max(5, len(losses) // 20)
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode="valid")
    smooth_steps = steps[window - 1:]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.plot(steps, losses, alpha=0.7, linewidth=0.5, color="steelblue",
            label="Raw Loss")
    ax.plot(smooth_steps, smoothed, linewidth=2, color="darkorange",
            label=f"Smoothed (window={window})")

    ax.set_title("Training Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "train_loss.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[charts] Saved {save_path} ({os.path.getsize(save_path)} bytes)")


def plot_scaling_curve(metrics_path: str, output_dir: str):
    """Plot test-time scaling curve from metrics.json.

    Reads metrics.json and plots accuracy vs thinking tokens, separating
    SFT model configs (solid blue, circle markers) from baseline configs
    (dashed red, square markers). Annotates each point with config name.

    Args:
        metrics_path: Path to metrics.json.
        output_dir: Directory to save scaling_curve.png.
    """
    if not os.path.exists(metrics_path):
        print(f"[charts] metrics.json not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    points = metrics.get("points", [])
    if not points:
        print("[charts] No data points in metrics.json -- skipping scaling_curve plot")
        return

    # Separate SFT and baseline using explicit type tag.
    # Backwards-compatible fallback: if type is missing, fall back to prefix check.
    sft_pts = []
    base_pts = []
    for p in points:
        t = p.get("type")
        if t == "sft":
            sft_pts.append(p)
        elif t == "baseline":
            base_pts.append(p)
        else:
            # Old metrics.json without type field -- use legacy prefix heuristic
            if p["config"].startswith("base_"):
                base_pts.append(p)
            else:
                sft_pts.append(p)

    # Sort by x (thinking tokens)
    sft_pts.sort(key=lambda p: p["x"])
    base_pts.sort(key=lambda p: p["x"])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    if sft_pts:
        sft_x = [p["x"] for p in sft_pts]
        sft_y = [p["y"] for p in sft_pts]
        ax.plot(sft_x, sft_y, "-o", color="steelblue", linewidth=2,
                markersize=8, label="SFT Model")
        for p in sft_pts:
            ax.annotate(p["config"], (p["x"], p["y"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)

    if base_pts:
        base_x = [p["x"] for p in base_pts]
        base_y = [p["y"] for p in base_pts]
        ax.plot(base_x, base_y, "--s", color="crimson", linewidth=2,
                markersize=8, label="Baseline")
        for p in base_pts:
            ax.annotate(p["config"], (p["x"], p["y"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)

    # Text box with summary stats (upper-left)
    control = metrics.get("control")
    scaling = metrics.get("scaling")
    performance = metrics.get("performance")
    control_pct = metrics.get("control_pct")

    text_lines = []
    if control is not None:
        if control_pct is not None:
            text_lines.append(f"Control: {control_pct}")
        else:
            text_lines.append(f"Control: {control:.4f}" if isinstance(control, float) else f"Control: {control}")
    if performance is not None:
        text_lines.append(f"Performance: {performance:.4f}" if isinstance(performance, float) else f"Performance: {performance}")
    if scaling is not None:
        text_lines.append(f"Scaling: {scaling:.6f}" if isinstance(scaling, float) else f"Scaling: {scaling}")

    if text_lines:
        textstr = "\n".join(text_lines)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_title("s1-t4: Test-Time Scaling Curve (MATH500 subset)")
    ax.set_xlabel("Thinking Tokens")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "scaling_curve.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[charts] Saved {save_path} ({os.path.getsize(save_path)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate charts for s1-t4 results"
    )
    parser.add_argument("--loss",
                        default="/content/s1-t4/results/train_loss.jsonl",
                        help="Path to train_loss.jsonl")
    parser.add_argument("--metrics",
                        default="/content/s1-t4/results/metrics.json",
                        help="Path to metrics.json")
    parser.add_argument("--output_dir",
                        default="/content/s1-t4/results/",
                        help="Output directory for charts")
    args = parser.parse_args()

    plot_train_loss(args.loss, args.output_dir)
    plot_scaling_curve(args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
