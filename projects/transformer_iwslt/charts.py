"""Post-hoc chart generation from downloaded metrics.jsonl files.

Reads output-{baseline,fixedpe,heads1}/metrics.jsonl + config.json,
produces 5 charts + results_summary.md.

Usage: python charts.py  (run locally after all experiments complete)
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent
OUTPUTS = {
    "baseline": PROJECT_DIR / "output-baseline",
    "fixed_pe": PROJECT_DIR / "output-fixedpe",
    "heads_1": PROJECT_DIR / "output-heads1",
}
CHARTS_DIR = PROJECT_DIR / "charts"
LABELS = {
    "baseline": "Baseline (8 heads, learned PE)",
    "fixed_pe": "Fixed Sinusoidal PE",
    "heads_1": "1 Attention Head",
}


def load_metrics(exp_id: str) -> list[dict]:
    path = OUTPUTS[exp_id] / "metrics.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {exp_id}")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    all_metrics = {}
    for exp_id in OUTPUTS:
        m = load_metrics(exp_id)
        if m:
            all_metrics[exp_id] = m

    if not all_metrics:
        print("No metrics found. Run experiments first.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- 1. Loss curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for exp_id, metrics in all_metrics.items():
        epochs = [m["epoch"] for m in metrics]
        ax1.plot(epochs, [m["train_loss"] for m in metrics],
                 label=LABELS[exp_id], linewidth=2)
        ax2.plot(epochs, [m["val_loss"] for m in metrics],
                 label=LABELS[exp_id], linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)
    print("Saved loss_curves.png")

    # --- 2. BLEU curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp_id, metrics in all_metrics.items():
        epochs = [m["epoch"] for m in metrics]
        ax.plot(epochs, [m["bleu"] for m in metrics],
                label=LABELS[exp_id], linewidth=2, marker="o")
    ax.set_title("SacreBLEU on IWSLT'14 De->En")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "bleu_curves.png", dpi=150)
    plt.close(fig)
    print("Saved bleu_curves.png")

    # --- 3. Ablation bars ---
    fig, ax = plt.subplots(figsize=(6, 5))
    exp_names = []
    final_bleus = []
    for exp_id, metrics in all_metrics.items():
        exp_names.append(LABELS[exp_id])
        final_bleus.append(max(m["bleu"] for m in metrics))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(range(len(exp_names)), final_bleus,
                  color=colors[:len(exp_names)])
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, fontsize=8, rotation=10)
    ax.set_ylabel("Best SacreBLEU")
    ax.set_title("Ablation: Final BLEU Comparison")
    for bar, bleu in zip(bars, final_bleus):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{bleu:.1f}",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ablation_bars.png", dpi=150)
    plt.close(fig)
    print("Saved ablation_bars.png")

    # --- 4. Attention heatmap (placeholder) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5,
            "Attention map requires\ntrained model checkpoint\n"
            "(run after final download)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="gray")
    ax.set_title("Attention Visualization (pending)")
    fig.savefig(CHARTS_DIR / "attention_heads.png", dpi=150)
    plt.close(fig)
    print("Saved attention_heads.png (placeholder)")

    # --- 5. Positional encoding comparison ---
    import torch
    from model import sinusoidal_pe

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sinusoidal PE similarity (always works)
    sin_pe = sinusoidal_pe(128, 512)[0].numpy()
    sim = sin_pe @ sin_pe.T
    im1 = ax1.imshow(sim, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax1.set_title("Sinusoidal PE — Cosine Similarity")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Position")
    fig.colorbar(im1, ax=ax1)

    # Learned PE placeholder
    ax2.text(0.5, 0.5,
             "Learned PE requires\ntrained baseline model\n"
             "(run after final download)",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=12, color="gray")
    ax2.set_title("Learned PE — Cosine Similarity (pending)")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "position_encoding.png", dpi=150)
    plt.close(fig)
    print("Saved position_encoding.png (sinusoidal side done, learned side pending)")

    # --- 6. Results summary ---
    lines = ["# Transformer IWSLT'14 De->En — Results Summary\n"]
    lines.append("| Experiment | Best BLEU | Final Train Loss | "
                 "Final Val Loss |")
    lines.append("|---|---|---|---|")
    for exp_id, metrics in all_metrics.items():
        best_bleu = max(m["bleu"] for m in metrics)
        final = metrics[-1]
        lines.append(
            f"| {LABELS[exp_id]} | {best_bleu:.1f} | "
            f"{final['train_loss']:.3f} | {final['val_loss']:.3f} |"
        )

    with open(CHARTS_DIR / "results_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("Saved results_summary.md")

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
