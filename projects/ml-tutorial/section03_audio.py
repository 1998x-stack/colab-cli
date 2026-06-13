"""Section 3 — Audio: Keyword spotting with Wav2Vec2 on Speech Commands.

Fine-tunes facebook/wav2vec2-base on 5 keyword classes, 500 samples each.
Saves model, metrics, and waveform visualization to <output_dir>/section03_audio/."""

import json
import os
import time
from datetime import datetime

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor, AutoModelForAudioClassification,
    Trainer, TrainingArguments,
)
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch


MODEL_NAME = "facebook/wav2vec2-base"
DATASET = "speech_commands"
KEYWORDS = ["yes", "no", "up", "down", "stop"]
SAMPLES_PER_KEYWORD = 500
TOTAL_TRAIN = len(KEYWORDS) * SAMPLES_PER_KEYWORD
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-4


def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] [AUDIO] {msg}", flush=True)


def filter_keywords(examples):
    return [lbl in KEYWORD_IDS for lbl in examples["label"]]


def remap_labels(examples):
    examples["label"] = [KEYWORD_IDS.index(lbl) for lbl in examples["label"]]
    return examples


def preprocess(batch, feature_extractor, sampling_rate=16000):
    audio_arrays = []
    for arr in batch["audio"]:
        audio_arrays.append(arr["array"])
    return feature_extractor(
        audio_arrays, sampling_rate=sampling_rate,
        padding=True, truncation=True, max_length=sampling_rate,
    )


def run(output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    section_dir = os.path.join(output_dir, "section03_audio")
    os.makedirs(section_dir, exist_ok=True)
    log(f"Device: {device}  Output: {section_dir}")

    # ── Load data ─────────────────────────────────────────────────
    log(f"Loading {DATASET}...")
    ds = load_dataset(DATASET)
    label_names = ds["train"].features["label"].names
    global KEYWORD_IDS
    KEYWORD_IDS = [label_names.index(k) for k in KEYWORDS]

    train_ds = ds["train"].filter(filter_keywords).map(remap_labels, batched=True)
    test_ds = ds["validation"].filter(filter_keywords).map(remap_labels, batched=True)
    # subsample
    def subsample(d, n_per):
        indices = []
        for cls in range(len(KEYWORDS)):
            cls_idxs = [i for i, lbl in enumerate(d["label"]) if lbl == cls][:n_per]
            indices.extend(cls_idxs)
        return d.select(indices)
    train_ds = subsample(train_ds, SAMPLES_PER_KEYWORD)
    test_ds = subsample(test_ds, SAMPLES_PER_KEYWORD // 5)  # 100 per keyword
    log(f"Train: {len(train_ds)}  Test: {len(test_ds)}  Keywords: {KEYWORDS}")

    # ── Preprocess ────────────────────────────────────────────────
    log(f"Loading feature extractor: {MODEL_NAME}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    train_ds = train_ds.with_transform(
        lambda b: preprocess(b, feature_extractor))
    test_ds = test_ds.with_transform(
        lambda b: preprocess(b, feature_extractor))

    # ── Train ─────────────────────────────────────────────────────
    log(f"Loading model: {MODEL_NAME}")
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=len(KEYWORDS), ignore_mismatched_sizes=True,
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
    feature_extractor.save_pretrained(os.path.join(section_dir, "model"))

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=KEYWORDS, yticklabels=KEYWORDS)
    plt.title("Audio — Speech Commands Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # ── Waveform visualization ────────────────────────────────────
    log("Generating waveform plot...")
    raw_test = ds["validation"].filter(filter_keywords).map(remap_labels, batched=True)
    raw_test = subsample(raw_test, SAMPLES_PER_KEYWORD // 5)

    fig, axes = plt.subplots(len(KEYWORDS), 1, figsize=(10, 2 * len(KEYWORDS)))
    for i, kw in enumerate(KEYWORDS):
        ax = axes[i] if len(KEYWORDS) > 1 else axes
        samples_for_kw = [j for j, lbl in enumerate(raw_test["label"]) if lbl == i]
        if samples_for_kw:
            idx = samples_for_kw[0]
            arr = raw_test[int(idx)]["audio"]["array"]
            sr = raw_test[int(idx)]["audio"]["sampling_rate"]
            t_axis = np.linspace(0, len(arr) / sr, len(arr))
            ax.plot(t_axis, arr, linewidth=0.5)
            ax.set_title(f"{kw} ({len(arr)/sr:.1f}s)")
            ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, "waveforms.png"), dpi=120)
    plt.close()

    # ── Per-keyword accuracy ──────────────────────────────────────
    per_keyword = {}
    for i, kw in enumerate(KEYWORDS):
        mask = y_true == i
        if mask.sum() > 0:
            per_keyword[kw] = round((y_pred[mask] == i).mean(), 4)

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "section": "03_audio",
        "task": "audio_classification",
        "model": MODEL_NAME,
        "dataset": DATASET,
        "keywords": KEYWORDS,
        "test_accuracy": round(test_acc, 4),
        "per_keyword_accuracy": per_keyword,
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
    run("/tmp/audio-test")
