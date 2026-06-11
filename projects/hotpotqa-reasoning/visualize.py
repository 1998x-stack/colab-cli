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
    per = data["per_example"]
    cot_lat = [r["cot_latency_s"] for r in per if r["cot_latency_s"]]
    react_lat = [r["react_latency_s"] for r in per if r["react_latency_s"]]

    fig, ax = plt.subplots(**STYLE)
    ax.boxplot([cot_lat, react_lat], labels=["CoT", "ReAct"], patch_artist=True,
               boxprops=dict(facecolor="#93c5fd"),
               medianprops=dict(color="#1e3a5f"))
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Per-Example Latency Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/latency_comparison.png")
    plt.close(fig)


def _token_efficiency(data: dict) -> None:
    fig, ax = plt.subplots(**STYLE)

    ax.scatter(
        data["cot"]["avg_tokens_per_example"], data["cot"]["exact_match"],
        s=200, label="CoT", color="#3b82f6", zorder=5,
    )
    ax.scatter(
        data["react"]["avg_tokens_per_example"], data["react"]["exact_match"],
        s=200, label="ReAct", color="#ef4444", zorder=5,
    )

    ax.set_xlabel("Avg Tokens per Example")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Token Efficiency: Accuracy vs Cost")
    ax.legend()
    ax.grid(alpha=0.3)
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
