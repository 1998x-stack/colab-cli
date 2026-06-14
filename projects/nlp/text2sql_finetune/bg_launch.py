"""Detached launcher: pip install + dataset prep + spawn training in background.
Exits in <30s. Training survives all WebSocket drops.
"""
import os, subprocess, sys, shutil

PROJECT_DIR = "/content/text2sql_finetune"

os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)

# Install deps
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "peft", "datasets", "accelerate", "torchao"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[bg] Deps installed")

# Clear stale HF cache
hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)

# Prepare dataset
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "train", "--max_examples", "600",
    "--output", f"{PROJECT_DIR}/data/all.pt",
])
import torch
all_data = torch.load(f"{PROJECT_DIR}/data/all.pt", weights_only=False)
torch.save(all_data[:500], f"{PROJECT_DIR}/data/train.pt")
torch.save(all_data[500:600], f"{PROJECT_DIR}/data/test.pt")
print(f"[bg] Dataset: 500 train, 100 test")

# Spawn training as detached background process
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open(f"{PROJECT_DIR}/logs/train.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"{PROJECT_DIR}/train.py",
         "--data_path", f"{PROJECT_DIR}/data/train.pt",
         "--output_dir", f"{PROJECT_DIR}/lora_weights",
         "--max_steps", "0", "--batch_size", "4", "--grad_accum", "2"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[bg] Training PID={proc.pid} — detached, survives exec timeout")
print(f"[bg] Monitor: tail /content/text2sql_finetune/logs/train.log")
print(f"[bg] Check: ls /content/text2sql_finetune/lora_weights/")
