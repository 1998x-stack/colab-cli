#!/usr/bin/env python3
"""Vision Transformer (ViT) on CIFAR-10 — multi-experiment, Kaggle-optimized.

Runs 3 configs sequentially: baseline, deeper, smallpatch.
Each experiment produces its own metrics.jsonl, charts.png, best_model.pt.
A comparison chart across all experiments is saved at the end.

Logs metrics.jsonl each epoch, saves charts per experiment.
"""
import os, sys, json, time, subprocess
from datetime import datetime

# ── GPU compatibility (handle P100 sm_60) ──────────────────────────
r = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True
)
gpu_name = r.stdout.strip()
print(f"GPU: {gpu_name}")

if "P100" in gpu_name:
    print("P100 detected — reinstalling PyTorch with CUDA 12.6 for sm_60 support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "torch", "torchvision",
        "--extra-index-url", "https://download.pytorch.org/whl/cu126"
    ], check=True, timeout=300)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

# ── Configs ───────────────────────────────────────────────────────
EXPERIMENTS = [
    {"slug": "vit-baseline",   "patch_size": 4, "depth": 6, "heads": 8, "dim": 256, "epochs": 12},
    {"slug": "vit-deeper",     "patch_size": 4, "depth": 10, "heads": 8, "dim": 256, "epochs": 12},
    {"slug": "vit-smallpatch", "patch_size": 2, "depth": 4, "heads": 6, "dim": 192, "epochs": 12},
]
BATCH_SIZE = 128
LR = 3e-4

# ── Paths ──────────────────────────────────────────────────────────
if os.path.exists("/kaggle/working/"):
    BASE_OUT = "/kaggle/working/output"
else:
    BASE_OUT = "./output"
os.makedirs(BASE_OUT, exist_ok=True)

# ── Data (shared across experiments) ───────────────────────────────
transform_train = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4471), (0.2470, 0.2435, 0.2616)),
])
transform_test = T.Compose([
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4471), (0.2470, 0.2435, 0.2616)),
])

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Data: {len(train_ds)} train, {len(test_ds)} test\n")

# ── ViT Model ───────────────────────────────────────────────────────
class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=256, depth=6, heads=8, mlp_ratio=4):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
            activation="gelu", batch_first=True, dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))

# ── Train one config ────────────────────────────────────────────────
def run_experiment(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg['slug']} — patch={cfg['patch_size']}, depth={cfg['depth']}, heads={cfg['heads']}, dim={cfg['dim']}")
    print(f"{'='*60}")

    model = ViT(
        img_size=32, patch_size=cfg["patch_size"], num_classes=10,
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"]
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ViT params: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    metrics = []
    best_acc = 0.0
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            preds = model(x).argmax(1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)
        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss = test_correct = test_total = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                test_loss += criterion(logits, y).item() * x.size(0)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += x.size(0)
        test_loss /= test_total
        test_acc = test_correct / test_total

        elapsed = time.time() - t_start
        entry = {
            "epoch": epoch, "experiment": cfg["slug"],
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4), "test_acc": round(test_acc, 4),
            "lr": scheduler.get_last_lr()[0], "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        metrics.append(entry)

        with open(f"{out_dir}/metrics.jsonl", "w") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")

        improved = "*" if test_acc > best_acc else ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{out_dir}/best_model.pt")

        print(f"  epoch {epoch:2d}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"test_loss={test_loss:.4f} acc={test_acc:.3f} | "
              f"{elapsed:.0f}s {improved}")

    cfg["best_acc"] = round(best_acc, 4)
    cfg["total_s"] = round(time.time() - t_start, 1)
    cfg["n_params_m"] = round(n_params, 1)
    print(f"  Done: best_acc={best_acc:.4f} in {cfg['total_s']:.0f}s")
    return metrics


# ── Run all experiments ────────────────────────────────────────────
ALL_START = time.time()
all_metrics = {}

for exp in EXPERIMENTS:
    out_dir = f"{BASE_OUT}/{exp['slug']}"
    m = run_experiment(exp, out_dir)
    all_metrics[exp["slug"]] = m

# ── Comparison chart ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for i, exp in enumerate(EXPERIMENTS):
    m = all_metrics[exp["slug"]]
    epochs = [e["epoch"] for e in m]
    ax = axes[i]
    ax.plot(epochs, [e["train_loss"] for e in m], "b-", label="train loss", alpha=0.7)
    ax.plot(epochs, [e["test_loss"] for e in m], "b--", label="test loss", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(epochs, [e["test_acc"] for e in m], "r-", label="test acc", linewidth=2)
    ax.set_title(f"{exp['slug']}\npatch={exp['patch_size']} depth={exp['depth']} best={exp['best_acc']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax2.set_ylabel("accuracy"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{BASE_OUT}/comparison.png", dpi=150)
plt.close()

# ── Per-experiment charts ──────────────────────────────────────────
for exp in EXPERIMENTS:
    m = all_metrics[exp["slug"]]
    epochs = [e["epoch"] for e in m]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [e["train_loss"] for e in m], marker="o", ms=3, label="train")
    ax1.plot(epochs, [e["test_loss"] for e in m], marker="o", ms=3, label="test")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Loss — {exp['slug']}")
    ax2.plot(epochs, [e["train_acc"] for e in m], marker="o", ms=3, label="train")
    ax2.plot(epochs, [e["test_acc"] for e in m], marker="o", ms=3, label="test")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Accuracy — best={exp['best_acc']}")
    plt.tight_layout()
    plt.savefig(f"{BASE_OUT}/{exp['slug']}/charts.png", dpi=120)
    plt.close()

# ── Summary ─────────────────────────────────────────────────────────
total_s = time.time() - ALL_START
print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*60}")
print(f"{'Experiment':20s}  {'Params':>7s}  {'Best Acc':>9s}  {'Time':>6s}")
print("-" * 50)
for exp in EXPERIMENTS:
    print(f"{exp['slug']:20s}  {exp['n_params_m']:5.1f}M  {exp['best_acc']:>9.4f}  {exp['total_s']:>5.0f}s")
print(f"Total: {total_s:.0f}s")
print(f"\nOutputs in {BASE_OUT}/")
for exp in EXPERIMENTS:
    print(f"  {exp['slug']}/  metrics.jsonl  charts.png  best_model.pt")
print(f"  comparison.png")

# Save overall summary
summary = {exp["slug"]: {"best_acc": exp["best_acc"], "params_m": exp["n_params_m"], "time_s": exp["total_s"], "config": {k: v for k, v in exp.items() if k not in ("best_acc", "total_s", "n_params_m")}} for exp in EXPERIMENTS}
summary["total_s"] = round(total_s, 1)
summary["gpu"] = gpu_name
with open(f"{BASE_OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
