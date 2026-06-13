"""Reusable training visualization — multi-panel PNGs for Colab/Kaggle projects.

Import guard: won't crash if matplotlib is missing.
Headless-safe: uses matplotlib.use("Agg") automatically.

Usage:
    from plot_utils import plot_loss_acc

    metrics = {
        "batch_losses": [1.2, 1.1, ...],
        "eval_losses": [(500, 1.05), (1000, 0.95)],
        "eval_accs": [(500, 0.45), (1000, 0.62)],
        "lr_values": [0.1, 0.099, ...],
    }
    plot_loss_acc(metrics, "/content/output/pngs/training_curves.png",
                  title="My Model — Dataset (SMALL config)", size_label="SMALL")
"""

import os

import numpy as np

_matplotlib_ok = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _matplotlib_ok = True
except Exception:
    pass


def _check():
    return _matplotlib_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Core: loss + accuracy + LR + distribution (classification / regression)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_acc(metrics, out_path, *, title="Training Curves", size_label=None,
                  window=100):
    """4-panel figure: loss curve, accuracy, LR schedule, loss distribution.

    Expects metrics dict with:
        batch_losses: list[float] — per-batch loss
        eval_losses: list[(batch, loss)] — optional
        eval_accs: list[(batch, acc)] — optional
        lr_values: list[float] — optional
    """
    if not _matplotlib_ok:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    suptitle = title
    if size_label:
        suptitle += f" ({size_label.upper()} config)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    bl = metrics.get("batch_losses", [])

    # --- Panel 1: Loss curve ---
    ax = axes[0, 0]
    if bl:
        ax.plot(bl, alpha=0.15, color="steelblue", linewidth=0.3, label="Batch loss")
        if len(bl) >= window:
            w = min(window, len(bl))
            smooth = np.convolve(bl, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(bl)), smooth, color="darkorange",
                    linewidth=1.5, label=f"avg{w}")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Eval accuracy ---
    ax = axes[0, 1]
    eval_accs = metrics.get("eval_accs", [])
    if eval_accs:
        xs, ys = zip(*eval_accs)
        ax.plot(xs, ys, "o-", color="mediumseagreen", linewidth=1.5, markersize=3)
        ax.set_ylabel("Accuracy")
        best = max(ys)
        ax.axhline(y=best, color="green", linestyle=":", alpha=0.7,
                   label=f"Best={best:.4f}")
        ax.legend(fontsize=8)
    elif metrics.get("eval_losses"):
        xs, ys = zip(*metrics["eval_losses"])
        ax.plot(xs, ys, "o-", color="coral", linewidth=1.5, markersize=3)
        ax.set_ylabel("Eval Loss")
    ax.set_xlabel("Batch")
    ax.set_title("Evaluation")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: LR schedule ---
    ax = axes[1, 0]
    lr_vals = metrics.get("lr_values", [])
    if lr_vals:
        ax.plot(lr_vals, color="crimson", linewidth=1.5)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
    else:
        ax.text(0.5, 0.5, "(no LR data)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Loss distribution ---
    ax = axes[1, 1]
    if bl:
        n_recent = min(2000, len(bl))
        recent = bl[-n_recent:]
        ax.hist(recent, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(x=np.mean(recent), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(recent):.3f}")
        ax.axvline(x=np.median(recent), color="orange", linestyle="--", linewidth=1.5,
                   label=f"median={np.median(recent):.3f}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.set_title(f"Loss Distribution (last {n_recent:,} batches)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# RL: reward + episode length + exploration + value distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rl_progress(metrics, out_path, *, title="RL Training", solved_threshold=None):
    """4-panel RL figure: reward curve, episode length, exploration decay, value dist.

    Expects metrics dict with:
        episode_rewards: list[float]
        episode_lengths: list[int]
        epsilon_values: list[float]  (or exploration values)
        q_values: list[float]  (mean Q per episode)
        avg_rewards: list[float]  (moving average)
    """
    if not _matplotlib_ok:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: Reward ---
    ax = axes[0, 0]
    rewards = metrics.get("episode_rewards", [])
    avg_r = metrics.get("avg_rewards", [])
    if rewards:
        ax.plot(rewards, alpha=0.2, color="steelblue", linewidth=0.5, label="Episode")
        if avg_r and len(avg_r) == len(rewards):
            ax.plot(avg_r, color="darkorange", linewidth=1.5, label="Moving avg")
        if solved_threshold:
            ax.axhline(y=solved_threshold, color="green", linestyle="--", alpha=0.7,
                       label=f"Solved={solved_threshold}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Episode length ---
    ax = axes[0, 1]
    lengths = metrics.get("episode_lengths", [])
    if lengths:
        ax.plot(lengths, color="mediumseagreen", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Exploration ---
    ax = axes[1, 0]
    eps_vals = metrics.get("epsilon_values", [])
    if eps_vals:
        ax.plot(eps_vals, color="crimson", linewidth=1.5)
        ax.set_ylabel("Epsilon / Entropy")
    ax.set_xlabel("Episode")
    ax.set_title("Exploration")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Value distribution ---
    ax = axes[1, 1]
    q_vals = metrics.get("q_values", [])
    if q_vals:
        ax.plot(q_vals, color="mediumpurple", linewidth=1.0, alpha=0.8)
        ax.set_ylabel("Mean Q")
    ax.set_xlabel("Episode")
    ax.set_title("Value Estimates")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal: single-panel loss plot (for quick checks)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss(metrics, out_path, *, title="Loss", window=100):
    """Single-panel loss plot — good for quick progress checks."""
    if not _matplotlib_ok:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    bl = metrics.get("batch_losses", [])
    if bl:
        ax.plot(bl, alpha=0.15, color="steelblue", linewidth=0.3)
        if len(bl) >= window:
            w = min(window, len(bl))
            smooth = np.convolve(bl, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(bl)), smooth, color="darkorange", linewidth=1.5,
                    label=f"avg{w}")
            ax.legend(fontsize=9)

    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
