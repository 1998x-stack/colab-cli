"""Colab bootstrap for text2sql_finetune.

Upload all source files, then run:
    colab exec -s <name> -f launch.py

Outputs go to /content/text2sql-finetune-output/
"""
import os
import subprocess
import sys
import shutil

PROJECT_DIR = "/content/text2sql_finetune"
OUTPUT_DIR = "/content/text2sql-finetune-output"
DEPS = ["peft", "datasets", "accelerate", "torch"]

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Clear stale HF cache ---
hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    print("[launch] Cleared HF datasets cache")

# --- Set env ---
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Prepare dataset ---
print("[launch] Preparing dataset...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "train", "--max_examples", "500",
    "--output", f"{PROJECT_DIR}/data/train.pt",
])
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "test", "--max_examples", "100",
    "--output", f"{PROJECT_DIR}/data/test.pt",
])
print("[launch] Dataset prepared")

# --- Train ---
print("[launch] Starting training...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/train.py",
    "--data_path", f"{PROJECT_DIR}/data/train.pt",
    "--output_dir", f"{PROJECT_DIR}/lora_weights",
    "--max_steps", "0",
])
print("[launch] Training complete")

# --- Evaluate ---
print("[launch] Running evaluation...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/evaluate.py",
    "--data_path", f"{PROJECT_DIR}/data/test.pt",
    "--lora_path", f"{PROJECT_DIR}/lora_weights",
    "--output", f"{OUTPUT_DIR}/eval_report.json",
])
print("[launch] Evaluation complete")

# --- Gather outputs ---
for src in [f"{PROJECT_DIR}/logs/train.log", f"{PROJECT_DIR}/logs/metrics.csv",
            f"{PROJECT_DIR}/logs/eval.log", f"{OUTPUT_DIR}/eval_report.json"]:
    if os.path.exists(src):
        dst = os.path.join(OUTPUT_DIR, os.path.relpath(src, PROJECT_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

# --- Tar outputs ---
subprocess.check_call([
    "tar", "-czf", "/content/text2sql-finetune-output.tar.gz",
    "-C", "/content", "text2sql-finetune-output",
])
print("[launch] DONE. Outputs at /content/text2sql-finetune-output.tar.gz")
print("[launch] Download: colab download -s <session> /content/text2sql-finetune-output.tar.gz")
