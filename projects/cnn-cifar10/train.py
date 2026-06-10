"""CNN image classifier on CIFAR-10 — fetches data, trains, logs metrics, visualizes."""

import json, os, time
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/cnn-cifar10-output"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
GRAD_CLIP = 1.0
PATIENCE = 5
NUM_CLASSES = 10

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)


# ── Data pipeline ─────────────────────────────────────────────────────────────
log("Loading CIFAR-10 dataset from HuggingFace...")
dataset = load_dataset("uoft-cs/cifar10")

train_transform = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(p=0.5),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]["img"]
        label = self.data[i]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


full_train = dataset["train"]
test_ds_raw = dataset["test"]

# Train/val split (80/20)
n_train = int(0.8 * len(full_train))
train_ds = CIFAR10Dataset(full_train.select(range(n_train)), train_transform)
val_ds = CIFAR10Dataset(full_train.select(range(n_train, len(full_train))), test_transform)
test_ds = CIFAR10Dataset(test_ds_raw, test_transform)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_ds, BATCH_SIZE, num_workers=2)

log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  Device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
    return total_loss / n, correct / n, y_true, y_pred


# ── Training ──────────────────────────────────────────────────────────────────
metrics_log = []
best_val_loss = float("inf")
patience_ctr = 0

log("Starting training...")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_n = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_correct += (model(x).argmax(1) == y).sum().item()  # cheap re-forward
        train_n += x.size(0)

    train_loss /= train_n
    train_acc = train_correct / train_n

    val_loss, val_acc, _, _ = evaluate(val_loader)
    scheduler.step(val_loss)

    entry = {
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "lr": optimizer.param_groups[0]["lr"],
    }
    metrics_log.append(entry)

    log(f"Epoch {epoch:2d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
        f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | lr: {entry['lr']:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
        log("  -> saved best model")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            log(f"Early stopping at epoch {epoch}")
            break

train_time = time.time() - t0
log(f"Training complete in {train_time/60:.1f}m. Best val_loss: {best_val_loss:.4f}")

# ── Test evaluation ───────────────────────────────────────────────────────────
log("Evaluating on test set...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model.pt"), map_location=DEVICE))
test_loss, test_acc, y_true, y_pred = evaluate(test_loader)
log(f"Test loss: {test_loss:.4f}  accuracy: {test_acc:.4f}")


# ── Visualizations ───────────────────────────────────────────────────────────
log("Generating plots...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = [m["epoch"] for m in metrics_log]

# Loss curve
axes[0].plot(epochs_range, [m["train_loss"] for m in metrics_log], "b-o", label="Train")
axes[0].plot(epochs_range, [m["val_loss"] for m in metrics_log], "r-o", label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("CIFAR-10 — Loss"); axes[0].legend(); axes[0].grid(True)

# Accuracy curve
axes[1].plot(epochs_range, [m["train_acc"] for m in metrics_log], "b-o", label="Train")
axes[1].plot(epochs_range, [m["val_acc"] for m in metrics_log], "r-o", label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("CIFAR-10 — Accuracy"); axes[1].legend(); axes[1].grid(True)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("CIFAR-10 — Confusion Matrix")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_accuracy_cm.png"), dpi=120)
plt.close()

# Sample predictions grid
log("Generating sample predictions grid...")
model.eval()
test_raw = dataset["test"]
indices = np.random.choice(len(test_raw), 25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, idx in enumerate(indices):
    ax = axes[i // 5][i % 5]
    sample = test_raw[int(idx)]
    img_t = test_transform(sample["img"]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_id = model(img_t).argmax(1).item()
    true_id = sample["label"]
    color = "green" if pred_id == true_id else "red"
    ax.imshow(sample["img"])
    ax.set_title(f"T: {CLASS_NAMES[true_id]}\nP: {CLASS_NAMES[pred_id]}",
                 fontsize=8, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=120)
plt.close()

# ── Save metrics ──────────────────────────────────────────────────────────────
class_accuracies = {}
for cls_id in range(NUM_CLASSES):
    mask = [t == cls_id for t in y_true]
    if sum(mask) > 0:
        cls_correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p == cls_id)
        class_accuracies[CLASS_NAMES[cls_id]] = round(cls_correct / sum(mask), 4)

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class_accuracy": class_accuracies,
        "train_time_seconds": round(train_time, 1),
        "device": str(DEVICE),
        "epochs": metrics_log,
    }, f, indent=2)

log(f"Done. All artifacts saved to {OUTPUT_DIR}/")
