"""Section 1 — NLP: Text classification with DistilBERT on AG News.

Fine-tunes distilbert-base-uncased on a 2000-sample subset, 3 epochs.
Saves model, metrics, and visualization to <output_dir>/section01_nlp/."""

import json, os, time
from datetime import datetime

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch


MODEL_NAME = "distilbert-base-uncased"
DATASET = "ag_news"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
TRAIN_SIZE = 2000
TEST_SIZE = 500
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-4


def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] [NLP] {msg}", flush=True)


def tokenize_fn(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=128)


def run(output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    section_dir = os.path.join(output_dir, "section01_nlp")
    os.makedirs(section_dir, exist_ok=True)
    log(f"Device: {device}  Output: {section_dir}")

    # ── Load data ─────────────────────────────────────────────────
    log(f"Loading {DATASET}...")
    ds = load_dataset(DATASET)
    train_ds = ds["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    test_ds = ds["test"].shuffle(seed=42).select(range(TEST_SIZE))
    log(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    # ── Tokenize ──────────────────────────────────────────────────
    log(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = train_ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    test_ds = test_ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ── Train ─────────────────────────────────────────────────────
    log(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES),
    ).to(device)

    args = TrainingArguments(
        output_dir=os.path.join(section_dir, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        use_cpu=False if device == "cuda" else True,
    )

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
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
    test_f1 = f1.compute(predictions=y_pred, references=y_true, average="macro")["f1"]
    log(f"Test accuracy: {test_acc:.4f}  F1: {test_f1:.4f}")

    # ── Save model ────────────────────────────────────────────────
    trainer.save_model(os.path.join(section_dir, "model"))
    tokenizer.save_pretrained(os.path.join(section_dir, "model"))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("NLP — AG News Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # ── Sample predictions ────────────────────────────────────────
    samples = test_ds.select(range(min(10, len(test_ds))))
    sample_text = []
    for i, s in enumerate(samples):
        true = CLASS_NAMES[s["label"]]
        pred = CLASS_NAMES[y_pred[i]]
        text = tokenizer.decode(s["input_ids"], skip_special_tokens=True)[:120]
        sample_text.append(f"TRUE: {true} | PRED: {pred}\n{text}\n")
    with open(os.path.join(section_dir, "sample_predictions.txt"), "w") as f:
        f.write("\n".join(sample_text))

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "section": "01_nlp",
        "task": "text_classification",
        "model": MODEL_NAME,
        "dataset": DATASET,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "train_time_seconds": round(train_time, 1),
        "device": device,
        "train_samples": TRAIN_SIZE,
        "test_samples": TEST_SIZE,
        "epochs": EPOCHS,
    }
    with open(os.path.join(section_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"Done in {train_time/60:.1f}m")
    return metrics


if __name__ == "__main__":
    run("/tmp/nlp-test")
