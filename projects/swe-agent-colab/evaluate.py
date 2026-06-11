#!/usr/bin/env python3
"""Generate charts and visualizations from metrics.json."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_metrics(path: str = "output/metrics.json") -> dict:
    with open(path) as f:
        return json.load(f)


def plot_results(metrics: dict, output_dir: str = "output"):
    out = Path(output_dir)
    tasks = metrics["per_task"]
    task_ids = [t["task_id"].split("__")[-1][:20] for t in tasks]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: Resolved per task
    colors = ["#22c55e" if t["resolved"] else "#ef4444" for t in tasks]
    axes[0].bar(range(len(tasks)), [1] * len(tasks), color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Task Resolution")
    axes[0].set_ylabel("Resolved")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["No", "Yes"])

    # Chart 2: Steps per task
    steps = [t["steps_taken"] for t in tasks]
    bars = axes[1].bar(range(len(tasks)), steps, color="#3b82f6", alpha=0.8)
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Steps Taken")
    axes[1].set_ylabel("Steps")
    for bar, s in zip(bars, steps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(s),
                     ha="center", fontsize=9)

    # Chart 3: Tokens per task
    prompt = [t["prompt_tokens"] for t in tasks]
    completion = [t["completion_tokens"] for t in tasks]
    x = range(len(tasks))
    width = 0.35
    axes[2].bar([i - width/2 for i in x], prompt, width, label="Prompt", color="#8b5cf6", alpha=0.8)
    axes[2].bar([i + width/2 for i in x], completion, width, label="Completion", color="#f59e0b", alpha=0.8)
    axes[2].set_xticks(range(len(tasks)))
    axes[2].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[2].set_title("Token Usage")
    axes[2].set_ylabel("Tokens")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out / "results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results.png")


def plot_timeline(metrics: dict, output_dir: str = "output"):
    """Plot per-task step duration timeline."""
    out = Path(output_dir)

    fig, ax = plt.subplots(figsize=(12, 5))

    for task in metrics["per_task"]:
        traj_path = out / f"trajectory_{task['task_id']}.json"
        if not traj_path.exists():
            continue
        with open(traj_path) as f:
            traj = json.load(f)
        steps = traj.get("trajectory", [])
        durations = [s.get("execution_time", 0) for s in steps if not s.get("done")]
        label = task["task_id"].split("__")[-1][:20]
        ax.plot(range(1, len(durations) + 1), durations, marker="o", label=label, markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Step Duration per Task")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timeline.png")


def plot_token_allocation(metrics: dict, output_dir: str = "output"):
    """Plot prompt vs completion token allocation per step for each task."""
    out = Path(output_dir)

    for task in metrics["per_task"]:
        traj_path = out / f"trajectory_{task['task_id']}.json"
        if not traj_path.exists():
            continue
        with open(traj_path) as f:
            traj = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 4))
        total_prompt = task["prompt_tokens"]
        total_completion = task["completion_tokens"]
        total = total_prompt + total_completion

        ax.pie(
            [total_prompt, total_completion],
            labels=["Prompt", "Completion"],
            colors=["#8b5cf6", "#f59e0b"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"Token Allocation — {task['task_id'].split('__')[-1][:30]}\n"
                     f"({total:,} total tokens)")

        plt.tight_layout()
        task_slug = task["task_id"].replace("/", "_")
        fig.savefig(out / f"token_allocation_{task_slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved token_allocation_{task_slug}.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="output/metrics.json")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if not Path(args.metrics).exists():
        print(f"Metrics file not found: {args.metrics}")
        sys.exit(1)

    metrics = load_metrics(args.metrics)
    plot_results(metrics, args.output_dir)
    plot_timeline(metrics, args.output_dir)
    plot_token_allocation(metrics, args.output_dir)
    print("All charts generated.")


if __name__ == "__main__":
    main()
