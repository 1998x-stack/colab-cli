#!/usr/bin/env python3
"""2-layer GCN on Cora, CiteSeer, PubMed — node classification.

Trains sequentially on all 3 citation networks. Per-dataset: CSV metrics,
PNG training curves, timestamped logs. Generates a comparison dashboard.

GPU-accelerated but small enough to run on CPU. Auto-detects Colab/Kaggle/local.
"""

import os
import time
from datetime import datetime

# ── Platform detection ────────────────────────────────────────────────────────
IN_KAGGLE = os.path.exists("/kaggle/working/")
IN_COLAB = os.path.exists("/content/")

if IN_KAGGLE:
    OUT_ROOT = "/kaggle/working/gnn-citation-output"
elif IN_COLAB:
    OUT_ROOT = "/content/gnn-citation-output"
else:
    OUT_ROOT = "./output/gnn-citation-output"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS = ["Cora", "CiteSeer", "PubMed"]
HIDDEN_DIM = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
SEED = 42

PRINT_EVERY = 10       # log per N epochs
PLOT_INTERVAL = 20     # save PNG every N epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── HF token ───────────────────────────────────────────────────────────────────
HF_TOKEN_PATH = "/content/.hf_token"
HF_TOKEN = None
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH) as f:
        HF_TOKEN = f.read().strip()

# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_dataset(name):
    """Load Cora/CiteSeer/PubMed — try HF datasets first, fall back to PyG Planetoid."""
    from torch_geometric.data import Data

    hf_id = f"gcaillaut/{name.lower()}"  # community-uploaded Cora/CiteSeer/PubMed

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset as hf_load
        hf_kwargs = {}
        if HF_TOKEN:
            hf_kwargs["token"] = HF_TOKEN
        hf_data = hf_load(hf_id, split="train", **hf_kwargs)
        print(f"[load] HF datasets: {hf_id} ({len(hf_data)} rows)")

        # HF dataset is a table of (node_id, features, label, edge_index)
        # Convert to PyG Data. Schema varies by uploader; handle common patterns.
        rows = list(hf_data)
        num_nodes = len(rows)

        # Collect features and labels
        x_list, y_list = [], []
        for r in rows:
            feat = r.get("features") or r.get("x") or r.get("node_features")
            label = r.get("label") or r.get("y")
            x_list.append(feat)
            y_list.append(label)
        x = torch.tensor(x_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.long)

        # Edges: some uploads include edge_index as (2, E), others as (E, 2) pairs
        if "edge_index" in rows[0]:
            ei = rows[0]["edge_index"]
            if isinstance(ei, list):
                edge_index = torch.tensor(ei, dtype=torch.long)
            else:
                edge_index = torch.tensor(ei, dtype=torch.long).t()
        else:
            raise ValueError("no edge_index in HF dataset")

        # Train/val/test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for i, r in enumerate(rows):
            split = r.get("split", r.get("mask", "train"))
            if split == "train" or split == 0:
                train_mask[i] = True
            elif split == "val" or split == 1:
                val_mask[i] = True
            else:
                test_mask[i] = True

        data = Data(x=x, y=y, edge_index=edge_index,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return data

    except Exception as e:
        print(f"[load] HF fallback ({e}), trying PyG Planetoid...")

    # Fallback: PyG Planetoid (downloads from GitHub, no auth needed)
    from torch_geometric.datasets import Planetoid
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Planetoid(tmp, name, split="public")
        data = dataset[0]
    return data

# ── GCN Model ─────────────────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ── Train one dataset ─────────────────────────────────────────────────────────
def train_one(name, out_dir):
    log_path = f"{out_dir}/train.log"
    csv_path = f"{out_dir}/metrics.csv"
    png_path = f"{out_dir}/training_curves.png"
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== GCN on {name} | device={device} ===")

    set_seed(SEED)
    data = load_dataset(name)
    data = data.to(device)

    num_classes = int(data.y.max().item()) + 1
    in_features = data.x.size(1)
    log(f"nodes={data.x.size(0)} edges={data.edge_index.size(1)//2} "
        f"features={in_features} classes={num_classes}")

    model = GCN(in_features, HIDDEN_DIM, num_classes, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            correct = (pred == data.y[mask]).sum().item()
            acc = correct / mask.sum().item()
        return acc, float(F.cross_entropy(out[mask], data.y[mask]))

    rows = []
    header = "epoch,loss,train_acc,val_acc,test_acc,elapsed_s"
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean().item()
        val_acc, _ = evaluate(data.val_mask)
        test_acc, _ = evaluate(data.test_mask)
        elapsed = time.time() - t_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        rows.append([epoch, round(loss.item(), 6), round(train_acc, 4),
                     round(val_acc, 4), round(test_acc, 4), round(elapsed, 2)])

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            log(f"Epoch {epoch:3d}/{EPOCHS} | loss={loss.item():.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"test_acc={test_acc:.4f} | elapsed={elapsed:.1f}s")

        if epoch % PLOT_INTERVAL == 0:
            save_png(name, rows, png_path)

    # Final saves
    save_csv(rows, csv_path, header)
    save_png(name, rows, png_path)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    final_test_acc, _ = evaluate(data.test_mask)
    log(f"=== DONE {name} | best_val={best_val_acc:.4f} "
        f"test={test_acc:.4f} elapsed={elapsed:.0f}s ===")

    return {"name": name, "rows": rows, "best_val": best_val_acc,
            "final_test": final_test_acc, "num_nodes": data.x.size(0),
            "num_edges": data.edge_index.size(1) // 2}

def save_csv(rows, path, header):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

def save_png(name, rows, path):
    if len(rows) < 2:
        return
    epochs_list = [r[0] for r in rows]
    losses = [r[1] for r in rows]
    train_accs = [r[2] for r in rows]
    val_accs = [r[3] for r in rows]
    test_accs = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"GCN on {name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs_list, losses, "tab:red", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_list, train_accs, "tab:blue", alpha=0.6, linewidth=1.0, label="Train")
    ax.plot(epochs_list, val_accs, "tab:green", linewidth=1.5, label="Val")
    ax.plot(epochs_list, test_accs, "tab:orange", linewidth=1.5, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    info = (f"Nodes: {rows[-1][0] if len(rows) > 0 else '?'}\n"
            f"Best Val Acc: {max(val_accs):.4f}\n"
            f"Final Test Acc: {test_accs[-1]:.4f}\n"
            f"Epochs: {len(rows)}")
    ax.text(0.1, 0.5, info, fontsize=12, fontfamily="monospace",
            verticalalignment="center", transform=ax.transAxes)
    ax.set_title("Summary")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

# ── Comparison dashboard ──────────────────────────────────────────────────────
def plot_comparison(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GCN — Citation Networks Comparison", fontsize=14, fontweight="bold")
    colors = {"Cora": "tab:blue", "CiteSeer": "tab:green", "PubMed": "tab:orange"}

    for res in results:
        name = res["name"]
        rows = res["rows"]
        epochs_list = [r[0] for r in rows]
        val_accs = [r[3] for r in rows]
        losses = [r[1] for r in rows]
        c = colors[name]

        axes[0, 0].plot(epochs_list, val_accs, color=c, linewidth=1.5, label=name)
        axes[0, 1].plot(epochs_list, losses, color=c, linewidth=1.2, label=name)

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Val Accuracy")
    axes[0, 0].set_title("Validation Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bar chart: final test accuracy
    names = [r["name"] for r in results]
    test_accs = [r["final_test"] for r in results]
    bars = axes[1, 0].bar(names, test_accs, color=[colors[n] for n in names], width=0.4)
    for bar, val in zip(bars, test_accs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f"{val:.4f}", ha="center", fontweight="bold")
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Final Test Accuracy")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Info table
    axes[1, 1].axis("off")
    info = "Dataset    Nodes   Edges   Test Acc\n"
    info += "-" * 42 + "\n"
    for res in results:
        info += f"{res['name']:<11} {res['num_nodes']:<7} {res['num_edges']:<7} {res['final_test']:.4f}\n"
    axes[1, 1].text(0.05, 0.95, info, fontsize=10, fontfamily="monospace",
                    verticalalignment="top", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_dashboard.png", dpi=150)
    plt.close()
    print(f"[comparison] saved to {out_dir}/comparison_dashboard.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"=== GCN — 3 Citation Networks ===  device={device}  {datetime.now()}")
    for d in DATASETS:
        print(f"  {d}")
    print()

    all_results = []
    for name in DATASETS:
        out_dir = f"{OUT_ROOT}/{name}"
        print(f"\n{'='*50}")
        print(f"  GCN on {name}")
        print(f"  Output: {out_dir}/")
        print(f"{'='*50}\n")
        result = train_one(name, out_dir)
        all_results.append(result)

    plot_comparison(all_results, f"{OUT_ROOT}/comparison")

    print(f"\n{'='*50}")
    print(f"  ALL DONE — {datetime.now()}")
    for r in all_results:
        print(f"  {r['name']:<10} test_acc={r['final_test']:.4f}  "
              f"nodes={r['num_nodes']}  edges={r['num_edges']}")
    print(f"\n  Output: {OUT_ROOT}/")
    print(f"  Comparison: {OUT_ROOT}/comparison/comparison_dashboard.png")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
