"""Matplotlib charts for CoT vs ReAct comparison."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/content/charts"
STYLE = {"figsize": (8, 5), "dpi": 120}


def generate_all(metrics_path: str = "/content/metrics.json") -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(metrics_path) as f:
        data = json.load(f)

    _accuracy_chart(data)
    _latency_chart(data)
    _token_efficiency(data)
    _steps_histogram(data)

    print(f"Charts saved to {OUTPUT_DIR}/")


def _accuracy_chart(data: dict) -> None:
    cot = data["cot"]
    react = data["react"]

    fig, ax = plt.subplots(**STYLE)
    x = np.arange(2)
    width = 0.3

    ax.bar(x - width / 2, [cot["exact_match"], cot["f1"]], width,
           label="CoT", color="#3b82f6")
    ax.bar(x + width / 2, [react["exact_match"], react["f1"]], width,
           label="ReAct", color="#ef4444")

    ax.set_ylabel("Score")
    ax.set_title("CoT vs ReAct: Accuracy on HotpotQA (200 examples)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Exact Match", "F1"])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    for bar in ax.containers:
        ax.bar_label(bar, fmt="%.3f", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")
    plt.close(fig)


def _latency_chart(data: dict) -> None:
    cot_lat = data["cot"]["avg_latency_s"]
    react_lat = data["react"]["avg_latency_s"]
    cot_wall = data["cot"]["total_wall_time_s"]
    react_wall = data["react"]["total_wall_time_s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # Per-example amortized latency
    x = np.arange(1)
    width = 0.3
    ax1.bar(x - width / 2, [cot_lat], width, label="CoT", color="#3b82f6")
    ax1.bar(x + width / 2, [react_lat], width, label="ReAct", color="#ef4444")
    ax1.set_ylabel("Seconds")
    ax1.set_title("Amortized Latency per Example")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Avg per example"])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for bar in ax1.containers:
        ax1.bar_label(bar, fmt="%.3f", fontsize=9)

    # Total wall time
    ax2.bar(x - width / 2, [cot_wall], width, label="CoT", color="#3b82f6")
    ax2.bar(x + width / 2, [react_wall], width, label="ReAct", color="#ef4444")
    ax2.set_ylabel("Seconds")
    ax2.set_title("Total Wall-Clock Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Total"])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bar in ax2.containers:
        ax2.bar_label(bar, fmt="%.1f", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/latency_comparison.png")
    plt.close(fig)


def _token_efficiency(data: dict) -> None:
    per = data["per_example"]

    cot_x = [r["cot_tokens"] for r in per]
    cot_y = [r["cot_em"] for r in per]
    react_x = [r["react_tokens"] for r in per]
    react_y = [r["react_em"] for r in per]

    fig, ax = plt.subplots(**STYLE)

    ax.scatter(cot_x, cot_y, s=30, label="CoT", color="#3b82f6",
               alpha=0.5, edgecolors="none")
    ax.scatter(react_x, react_y, s=30, label="ReAct", color="#ef4444",
               alpha=0.5, edgecolors="none")

    # Averages as larger markers
    ax.scatter(
        data["cot"]["avg_tokens_per_example"], data["cot"]["exact_match"],
        s=200, color="#3b82f6", edgecolors="white", linewidths=1.5, zorder=5,
    )
    ax.scatter(
        data["react"]["avg_tokens_per_example"], data["react"]["exact_match"],
        s=200, color="#ef4444", edgecolors="white", linewidths=1.5, zorder=5,
    )

    ax.set_xlabel("Tokens per Example")
    ax.set_ylabel("Exact Match (1=correct, 0=wrong)")
    ax.set_title("Token Efficiency: Per-Example Accuracy vs Cost")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/token_efficiency.png")
    plt.close(fig)


def _steps_histogram(data: dict) -> None:
    per = data["per_example"]
    steps_correct = [r["react_steps"] for r in per if r["react_em"] == 1]
    steps_wrong = [r["react_steps"] for r in per if r["react_em"] == 0]

    fig, ax = plt.subplots(**STYLE)
    bins = np.arange(0.5, 6.5, 1)
    ax.hist([steps_correct, steps_wrong], bins=bins, label=["Correct", "Incorrect"],
            color=["#22c55e", "#f87171"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of ReAct Steps")
    ax.set_ylabel("Count")
    ax.set_title("ReAct Steps Distribution (Correct vs Incorrect)")
    ax.set_xticks(range(1, 6))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/react_steps.png")
    plt.close(fig)


if __name__ == "__main__":
    generate_all()
