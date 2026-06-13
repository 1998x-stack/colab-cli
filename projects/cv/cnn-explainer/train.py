"""CNN Explainer — train a CNN on CIFAR-10 and generate explainability visualizations.

Techniques: Grad-CAM, Saliency Maps, Guided Backprop, Integrated Gradients, Feature Maps.

Usage:
    python train.py
    python train.py --epochs 20 --lr 0.0005 --batch-size 64
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CNN Explainer — CIFAR-10")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="/content/cnn-explainer-output")
    p.add_argument("--num-explain", type=int, default=16, help="samples for explainability dashboard")
    p.add_argument("--ig-steps", type=int, default=20, help="integrated gradients steps")
    return p.parse_args()


CFG = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10

os.makedirs(CFG.output_dir, exist_ok=True)
for sub in ["logs", "pngs"]:
    os.makedirs(os.path.join(CFG.output_dir, sub), exist_ok=True)


# ── Logging ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)


def write_csv(path: str, row: dict, first: bool = False):
    mode = "w" if first else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


# ── Data ────────────────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
eval_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
# Unnormalized transform for visualization
viz_tf = T.Compose([T.ToTensor()])


class CIFAR10(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]["img"]
        y = self.data[i]["label"]
        if self.transform:
            x = self.transform(x)
        return x, y


def build_loaders():
    ds = load_dataset("uoft-cs/cifar10")
    full_train = ds["train"]
    test_raw = ds["test"]

    n_train = int(0.8 * len(full_train))

    train_ds = CIFAR10(full_train.select(range(n_train)), train_tf)
    val_ds = CIFAR10(full_train.select(range(n_train, len(full_train))), eval_tf)
    test_ds = CIFAR10(test_raw, eval_tf)
    test_ds_viz = CIFAR10(test_raw, viz_tf)  # unnormalized for viz

    train_ldr = DataLoader(train_ds, CFG.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ldr = DataLoader(val_ds, CFG.batch_size, num_workers=2, pin_memory=True)
    test_ldr = DataLoader(test_ds, CFG.batch_size, num_workers=2, pin_memory=True)

    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  Device: {DEVICE}")
    return train_ldr, val_ldr, test_ldr, test_ds_viz, test_raw


# ── Model ───────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)  # last conv — target for Grad-CAM
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

    @property
    def last_conv(self):
        return self.block3.conv


def build_model():
    m = CNN().to(DEVICE)
    n = sum(p.numel() for p in m.parameters())
    log(f"Parameters: {n:,}")
    return m


# ── Training helpers ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        total_loss += F.cross_entropy(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(out.argmax(1).cpu().tolist())
    return total_loss / n, correct / n, y_true, y_pred


# ── Explainability ──────────────────────────────────────────────────────────────

def grad_cam(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap from last conv layer. Returns (H, W) numpy array."""
    activations = []
    gradients = []

    def fwd_hook(m, inp, out):
        activations.append(out)

    def bwd_hook(m, grad_in, grad_out):
        gradients.append(grad_out[0])

    target = model.last_conv
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    out = model(x)
    model.zero_grad()
    out[0, class_idx].backward(retain_graph=True)

    pooled = gradients[0].mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (pooled * activations[0]).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.squeeze().detach().cpu().numpy()


def saliency_map(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Simple gradient saliency. Returns (H, W) numpy array."""
    x_in = x.clone().detach().requires_grad_(True)
    out = model(x_in)
    model.zero_grad()
    out[0, class_idx].backward()
    sal = x_in.grad.abs().max(dim=1)[0]  # max magnitude across RGB
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.squeeze().detach().cpu().numpy()


def integrated_gradients(model, x: torch.Tensor, class_idx: int, steps: int = 20) -> np.ndarray:
    """Integrated Gradients from black baseline (zero image). Returns (H, W) numpy."""
    baseline = torch.zeros_like(x)
    ig = torch.zeros_like(x)
    for alpha in np.linspace(0, 1, steps):
        interp = baseline + alpha * (x - baseline)
        interp = interp.clone().detach().requires_grad_(True)
        out = model(interp)
        model.zero_grad()
        out[0, class_idx].backward(retain_graph=True)
        ig += interp.grad
    ig = (ig * (x - baseline)).abs().max(dim=1)[0]
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
    return ig.squeeze().detach().cpu().numpy()


def apply_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a colormap heatmap on an image. Both (H, W, 3) uint8 and (H, W) float."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float
    overlay = alpha * colored + (1 - alpha) * img_np.astype(np.float32) / 255.0
    return (overlay * 255).astype(np.uint8)


# ── Visualizations ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def save_training_curves(metrics: list, output_dir: str):
    """4-panel figure: loss, accuracy, LR, confusion matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    epochs = [m["epoch"] for m in metrics]

    # Loss
    ax = axes[0][0]
    ax.plot(epochs, [m["train_loss"] for m in metrics], "b-o", ms=4, label="Train")
    ax.plot(epochs, [m["val_loss"] for m in metrics], "r-o", ms=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0][1]
    ax.plot(epochs, [m["train_acc"] for m in metrics], "b-o", ms=4, label="Train")
    ax.plot(epochs, [m["val_acc"] for m in metrics], "r-o", ms=4, label="Val")
    ax.axhline(y=0.90, color="gray", ls="--", alpha=0.5, label="90% baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1][0]
    ax.plot(epochs, [m["lr"] for m in metrics], "g-s", ms=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Confusion matrix placeholder (filled after test eval)
    axes[1][1].text(0.5, 0.5, "Confusion Matrix\n(computed after training)",
                    ha="center", va="center", transform=axes[1][1].transAxes, fontsize=12)
    axes[1][1].set_title("Confusion Matrix")

    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    log(f"Saved training curves → {path}")


def save_confusion_matrix(y_true: list, y_pred: list, output_dir: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CIFAR-10 — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    log(f"Saved confusion matrix → {path}")


def save_explainer_dashboard(model, test_ds_viz, test_raw, num: int, output_dir: str):
    """Multi-panel dashboard: Grad-CAM, Saliency, Integrated Gradients per sample."""
    model.eval()
    n = min(num, len(test_raw))
    indices = np.random.choice(len(test_raw), n, replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Input", "Grad-CAM", "Saliency Map", "Int. Gradients"]
    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        idx = int(idx)
        sample = test_raw[idx]
        img_pil = sample["img"]
        true_label = CLASS_NAMES[sample["label"]]

        # Prepare tensors
        img_viz = viz_tf(img_pil).unsqueeze(0).to(DEVICE)  # unnormalized for display
        img_eval = eval_tf(img_pil).unsqueeze(0).to(DEVICE)  # normalized for model

        with torch.no_grad():
            out = model(img_eval)
            pred_id = out.argmax(1).item()

        pred_label = CLASS_NAMES[pred_id]
        correct = pred_id == sample["label"]
        title_color = "green" if correct else "red"

        # Column 0: Input
        img_np = img_viz.squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[row][0].imshow(img_np)
        axes[row][0].set_ylabel(f"T: {true_label}\nP: {pred_label}", fontsize=8, color=title_color)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])

        # Compute explainability maps (use predicted class)
        cam = grad_cam(model, img_eval, pred_id)
        sal = saliency_map(model, img_eval, pred_id)
        ig = integrated_gradients(model, img_eval, pred_id, steps=CFG.ig_steps)

        # Column 1: Grad-CAM
        axes[row][1].imshow(img_np)
        axes[row][1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row][1].set_xticks([])
        axes[row][1].set_yticks([])

        # Column 2: Saliency
        axes[row][2].imshow(sal, cmap="hot")
        axes[row][2].set_xticks([])
        axes[row][2].set_yticks([])

        # Column 3: Integrated Gradients
        axes[row][3].imshow(ig, cmap="hot")
        axes[row][3].set_xticks([])
        axes[row][3].set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "explainer_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved explainer dashboard → {path}")


def save_feature_maps(model, test_ds_viz, test_raw, output_dir: str):
    """Top-8 activating images for 8 random filters from each conv layer."""
    model.eval()

    # Get one batch of test images
    indices = np.random.choice(len(test_raw), 64, replace=False)
    imgs = []
    for idx in indices:
        sample = test_raw[int(idx)]
        imgs.append(eval_tf(sample["img"]))
    x = torch.stack(imgs).to(DEVICE)

    # Hook each block's conv output
    activations = {}
    def make_hook(name):
        def hook(m, inp, out):
            activations[name] = out
        return hook

    hooks = [
        model.block1.conv.register_forward_hook(make_hook("block1")),
        model.block2.conv.register_forward_hook(make_hook("block2")),
        model.block3.conv.register_forward_hook(make_hook("block3")),
    ]

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    fig, all_axes = plt.subplots(3, 8, figsize=(16, 7))
    for layer_idx, (name, acts) in enumerate(activations.items()):
        n_filters = acts.shape[1]
        selected = np.random.choice(n_filters, min(8, n_filters), replace=False)
        for col, fidx in enumerate(selected):
            ax = all_axes[layer_idx][col]
            # Find top-activating image for this filter
            filter_acts = acts[:, fidx, :, :].mean(dim=(1, 2))  # (64,)
            best_img_idx = filter_acts.argmax().item()
            best_img = imgs[best_img_idx]
            # Denormalize for display
            img_np = best_img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{name}\nch={acts.shape[1]}", fontsize=9)
            ax.set_title(f"F#{fidx}", fontsize=8)

    plt.suptitle("Feature Maps — top-activating image per filter", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "feature_maps.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"Saved feature maps → {path}")


def save_sample_predictions(model, test_ds_viz, test_raw, output_dir: str):
    """5x5 grid of predictions with correct/incorrect coloring."""
    model.eval()
    indices = np.random.choice(len(test_raw), 25, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, idx in enumerate(indices):
        ax = axes[i // 5][i % 5]
        sample = test_raw[int(idx)]
        img_t = eval_tf(sample["img"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_id = model(img_t).argmax(1).item()
        true_id = sample["label"]
        color = "green" if pred_id == true_id else "red"
        ax.imshow(sample["img"])
        ax.set_title(f"T: {CLASS_NAMES[true_id]}\nP: {CLASS_NAMES[pred_id]}",
                     fontsize=7, color=color)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "sample_predictions.png")
    plt.savefig(path, dpi=120)
    plt.close()
    log(f"Saved sample predictions → {path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    log(f"CNN Explainer — CIFAR-10  |  epochs={CFG.epochs}  batch={CFG.batch_size}  "
        f"lr={CFG.lr}  device={DEVICE}")
    log(f"Output: {CFG.output_dir}")

    train_ldr, val_ldr, test_ldr, test_ds_viz, test_raw = build_loaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    csv_path = os.path.join(CFG.output_dir, "metrics.csv")
    best_val_loss = float("inf")
    metrics_log = []

    log("Starting training...")
    t_start = time.time()

    for epoch in range(1, CFG.epochs + 1):
        t_ep = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for x, y in train_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            # Re-forward for accuracy (cheaper than storing all logits)
            with torch.no_grad():
                out = model(x)
                train_correct += (out.argmax(1) == y).sum().item()
            train_n += x.size(0)

        train_loss /= train_n
        train_acc = train_correct / train_n

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_ldr)
        scheduler.step()

        elapsed = time.time() - t_ep

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "elapsed_s": round(elapsed, 1),
        }
        metrics_log.append(entry)
        write_csv(csv_path, entry, first=(epoch == 1))

        log(f"Epoch {epoch:2d} | train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | lr: {entry['lr']:.2e} | {elapsed:.0f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CFG.output_dir, "model.pt"))
            log("  -> saved best model")

        # Mid-training curves (every 2 epochs)
        if epoch % 2 == 0:
            save_training_curves(metrics_log, CFG.output_dir)

    train_time = time.time() - t_start
    log(f"Training complete in {train_time/60:.1f}m. Best val_loss: {best_val_loss:.4f}")

    # ── Test evaluation ──────────────────────────────────────────────────────
    log("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(CFG.output_dir, "model.pt"), map_location=DEVICE, weights_only=True))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_ldr)
    log(f"Test loss: {test_loss:.4f}  accuracy: {test_acc:.4f}")

    # ── Visualizations ───────────────────────────────────────────────────────
    log("Generating training curves...")
    save_training_curves(metrics_log, CFG.output_dir)
    save_confusion_matrix(y_true, y_pred, CFG.output_dir)
    save_sample_predictions(model, test_ds_viz, test_raw, CFG.output_dir)

    log("Generating explainability dashboard...")
    save_explainer_dashboard(model, test_ds_viz, test_raw, CFG.num_explain, CFG.output_dir)
    save_feature_maps(model, test_ds_viz, test_raw, CFG.output_dir)

    # ── Per-class accuracy ───────────────────────────────────────────────────
    cls_acc = {}
    for cls_id in range(NUM_CLASSES):
        mask = [t == cls_id for t in y_true]
        if sum(mask) > 0:
            cls_correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p == cls_id)
            cls_acc[CLASS_NAMES[cls_id]] = round(cls_correct / sum(mask), 4)

    # ── Save summary ─────────────────────────────────────────────────────────
    summary = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class_accuracy": cls_acc,
        "train_time_s": round(train_time, 1),
        "best_val_loss": round(best_val_loss, 4),
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": str(DEVICE),
        "config": vars(CFG),
        "epochs": metrics_log,
    }
    with open(os.path.join(CFG.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Done. All artifacts in {CFG.output_dir}/")
    log("  metrics.csv  |  model.pt  |  summary.json")
    log("  pngs/training_curves.png  |  pngs/confusion_matrix.png")
    log("  pngs/explainer_dashboard.png  |  pngs/feature_maps.png  |  pngs/sample_predictions.png")


if __name__ == "__main__":
    main()
