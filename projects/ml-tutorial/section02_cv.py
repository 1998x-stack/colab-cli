"""Section 2 — CV: Image classification with ViT on Food101 (5-class subset).

Fine-tunes google/vit-base-patch16-224 on 5 food classes, 250 samples each.
Saves model, metrics, and prediction grid to <output_dir>/section02_cv/."""

import json, os, time
from datetime import datetime

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    Trainer, TrainingArguments,
)
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch


MODEL_NAME = "google/vit-base-patch16-224"
DATASET = "food101"
# subset: pick 5 well-separated classes
CLASS_SUBSET = ["pizza", "sushi", "hamburger", "ice_cream", "waffles"]
CLASS_IDS = None  # resolved after loading
SAMPLES_PER_CLASS = 250
TOTAL_TRAIN = len(CLASS_SUBSET) * SAMPLES_PER_CLASS  # 1250
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-4


def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] [CV] {msg}", flush=True)


def filter_classes(examples):
    return [lbl in CLASS_IDS for lbl in examples["label"]]


def remap_labels(examples):
    examples["label"] = [CLASS_IDS.index(lbl) for lbl in examples["label"]]
    return examples


def transform_fn(batch, processor):
    images = [img.convert("RGB") for img in batch["image"]]
    return processor(images=images)


def run(output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    section_dir = os.path.join(output_dir, "section02_cv")
    os.makedirs(section_dir, exist_ok=True)
    log(f"Device: {device}  Output: {section_dir}")

    # ── Load data ─────────────────────────────────────────────────
    log(f"Loading {DATASET}...")
    ds = load_dataset(DATASET)
    # resolve class names to ids
    global CLASS_IDS
    label_names = ds["train"].features["label"].names
    CLASS_IDS = [label_names.index(c) for c in CLASS_SUBSET]
    class_names = CLASS_SUBSET

    train_ds = ds["train"].filter(filter_classes).map(remap_labels, batched=True)
    test_ds = ds["validation"].filter(filter_classes).map(remap_labels, batched=True)
    # subsample
    def subsample(d, n_per_class):
        indices = []
        for cls in range(len(CLASS_SUBSET)):
            cls_idxs = [i for i, lbl in enumerate(d["label"]) if lbl == cls][:n_per_class]
            indices.extend(cls_idxs)
        return d.select(indices)
    train_ds = subsample(train_ds, SAMPLES_PER_CLASS)
    test_ds = subsample(test_ds, SAMPLES_PER_CLASS // 5)  # 50 per class
    log(f"Train: {len(train_ds)}  Test: {len(test_ds)}  Classes: {class_names}")

    # ── Preprocess ────────────────────────────────────────────────
    log(f"Loading processor: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    train_ds = train_ds.with_transform(lambda b: transform_fn(b, processor))
    test_ds = test_ds.with_transform(lambda b: transform_fn(b, processor))

    # ── Train ─────────────────────────────────────────────────────
    log(f"Loading model: {MODEL_NAME}")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(class_names), ignore_mismatched_sizes=True,
    ).to(device)

    args = TrainingArguments(
        output_dir=os.path.join(section_dir, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=False if device == "cuda" else True,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    log("Starting training...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # ── Evaluate ──────────────────────────────────────────────────
    log("Evaluating...")
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    test_acc = accuracy.compute(predictions=y_pred, references=y_true)["accuracy"]
    log(f"Test accuracy: {test_acc:.4f}")

    # ── Save model ────────────────────────────────────────────────
    trainer.save_model(os.path.join(section_dir, "model"))
    processor.save_pretrained(os.path.join(section_dir, "model"))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("CV — Food101 Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # ── Sample predictions grid ───────────────────────────────────
    log("Generating prediction grid...")
    n_show = min(10, len(test_ds))
    raw_test = ds["validation"].filter(filter_classes).map(remap_labels, batched=True)
    raw_test = subsample(raw_test, SAMPLES_PER_CLASS // 5)
    show_indices = np.random.choice(len(raw_test), n_show, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, idx in enumerate(show_indices):
        ax = axes[i // 5][i % 5]
        sample = raw_test[int(idx)]
        img = sample["image"].convert("RGB")
        true_id = sample["label"]
        pred_id = y_pred[int(idx)]
        color = "green" if pred_id == true_id else "red"
        ax.imshow(img)
        ax.set_title(f"T: {class_names[true_id]}\nP: {class_names[pred_id]}",
                     fontsize=8, color=color)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "sample_predictions.png"), dpi=120)
    plt.close()

    # ── Per-class accuracy ────────────────────────────────────────
    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[name] = round((y_pred[mask] == i).mean(), 4)

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "section": "02_cv",
        "task": "image_classification",
        "model": MODEL_NAME,
        "dataset": DATASET,
        "classes": class_names,
        "test_accuracy": round(test_acc, 4),
        "per_class_accuracy": per_class,
        "train_time_seconds": round(train_time, 1),
        "device": device,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "epochs": EPOCHS,
    }
    with open(os.path.join(section_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"Done in {train_time/60:.1f}m")
    return metrics


if __name__ == "__main__":
    run("/tmp/cv-test")
