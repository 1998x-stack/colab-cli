"""LSTM sentiment classifier on IMDB — fetches data, trains, logs metrics, visualizes."""

import json, os, sys, time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/rnn-imdb-output"
MAX_VOCAB = 25000
MAX_LEN = 500
EMBED_DIM = 256
HIDDEN_DIM = 128
DROPOUT = 0.5
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
GRAD_CLIP = 1.0
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)


# ── Data pipeline ─────────────────────────────────────────────────────────────
log("Loading IMDB dataset from HuggingFace...")
dataset = load_dataset("stanfordnlp/imdb")
train_raw = dataset["train"]
test_raw = dataset["test"]

log("Building vocabulary...")
counter = Counter()
for ex in train_raw:
    for word in ex["text"].lower().split():
        counter[word] += 1

vocab = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(MAX_VOCAB)]
word2idx = {w: i for i, w in enumerate(vocab)}


def tokenize(text):
    ids = [word2idx.get(w, 1) for w in text.lower().split()[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return tokenize(self.data[i]["text"]), torch.tensor(self.data[i]["label"], dtype=torch.float32)


# Train/val split (80/20)
n_train = int(0.8 * len(train_raw))
train_ds = IMDBDataset(train_raw.select(range(n_train)))
val_ds = IMDBDataset(train_raw.select(range(n_train, len(train_raw))))
test_ds = IMDBDataset(test_raw)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE)
test_loader = DataLoader(test_ds, BATCH_SIZE)

log(f"Vocab size: {len(vocab)}, train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}, device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(HIDDEN_DIM * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = self.dropout(h)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        return self.sigmoid(self.fc2(h)).squeeze(-1)


model = LSTMClassifier().to(DEVICE)
criterion = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)


def compute_f1(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


# ── Training ──────────────────────────────────────────────────────────────────
metrics_log = []
best_val_loss = float("inf")
patience_ctr = 0

log("Starting training...")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss, train_correct, train_n = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_correct += ((pred > 0.5).float() == y).sum().item()
        train_n += x.size(0)

    train_loss /= train_n
    train_acc = train_correct / train_n

    # Val
    model.eval()
    val_loss, val_correct, val_n = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)
            val_correct += ((pred > 0.5).float() == y).sum().item()
            val_n += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend((pred.cpu() > 0.5).int().tolist())

    val_loss /= val_n
    val_acc = val_correct / val_n
    val_f1 = compute_f1(torch.tensor(y_true), torch.tensor(y_pred))

    scheduler.step(val_loss)

    entry = {
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_f1": round(val_f1, 4),
        "lr": optimizer.param_groups[0]["lr"],
    }
    metrics_log.append(entry)

    log(f"Epoch {epoch:2d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
        f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_f1: {val_f1:.4f} | lr: {entry['lr']:.2e}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        torch.save({"model": model.state_dict(), "vocab": vocab, "word2idx": word2idx},
                   os.path.join(OUTPUT_DIR, "model.pt"))
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
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "model.pt"), map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

test_loss, test_correct, test_n = 0, 0, 0
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)
        test_loss += loss.item() * x.size(0)
        test_correct += ((pred > 0.5).float() == y).sum().item()
        test_n += x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend((pred.cpu() > 0.5).int().tolist())

test_acc = test_correct / test_n
test_f1 = compute_f1(torch.tensor(y_true), torch.tensor(y_pred))
log(f"Test loss: {test_loss/test_n:.4f}  accuracy: {test_acc:.4f}  F1: {test_f1:.4f}")

# ── Sample predictions ────────────────────────────────────────────────────────
idx2word = {i: w for w, i in word2idx.items()}
samples = []
for i in range(min(20, len(test_raw))):
    review = " ".join([idx2word.get(tok, "<unk>") for tok in test_ds[i][0].tolist() if tok != 0])
    samples.append({
        "text": review[:200],
        "true": int(y_true[i]),
        "pred": int(y_pred[i]),
    })

with open(os.path.join(OUTPUT_DIR, "predictions.txt"), "w") as f:
    for s in samples:
        f.write(f"TRUE: {'POS' if s['true'] else 'NEG'} | PRED: {'POS' if s['pred'] else 'NEG'} | {s['text']}...\n\n")

# ── Visualizations ────────────────────────────────────────────────────────────
log("Generating plots...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

epochs_range = [m["epoch"] for m in metrics_log]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(epochs_range, [m["train_loss"] for m in metrics_log], "b-o", label="Train")
axes[0].plot(epochs_range, [m["val_loss"] for m in metrics_log], "r-o", label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("IMDB — Loss"); axes[0].legend(); axes[0].grid(True)

# Accuracy curve
axes[1].plot(epochs_range, [m["train_acc"] for m in metrics_log], "b-o", label="Train")
axes[1].plot(epochs_range, [m["val_acc"] for m in metrics_log], "r-o", label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("IMDB — Accuracy"); axes[1].legend(); axes[1].grid(True)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("IMDB — Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_accuracy_cm.png"), dpi=120)
plt.close()

# F1 curve (separate plot)
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, [m["val_f1"] for m in metrics_log], "g-o")
plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.title("IMDB — Validation F1")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_curve.png"), dpi=120)
plt.close()

# ── Save metrics ──────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({
        "test_loss": round(test_loss / test_n, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "train_time_seconds": round(train_time, 1),
        "device": str(DEVICE),
        "epochs": metrics_log,
    }, f, indent=2)

log(f"Done. All artifacts saved to {OUTPUT_DIR}/")
