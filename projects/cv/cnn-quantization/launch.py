"""
CNN Quantization — Colab bootstrap.
Upload train.py to VM, install deps, spawn detached training.
Usage: colab exec -f launch.py --timeout 120
"""
import subprocess
import sys
import os
from pathlib import Path

SCRIPT = os.environ.get("LAUNCH_SCRIPT", "train.py")
DEPS = os.environ.get("LAUNCH_DEPS", "torchvision,pandas,matplotlib")
LOG = os.environ.get("LAUNCH_LOG", "/content/cnn-quantization-output/logs/train.log")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EPOCHS = os.environ.get("LAUNCH_EPOCHS", "10")
BATCH_SIZE = os.environ.get("LAUNCH_BATCH", "128")

script_path = Path(f"/content/{SCRIPT}")
log_dir = Path(LOG).parent
log_dir.mkdir(parents=True, exist_ok=True)

# Install deps
deps = [d.strip() for d in DEPS.split(",") if d.strip()]
print(f"Installing: {deps}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + deps, timeout=120)

# Verify GPU
subprocess.run([sys.executable, "-c",
    "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"],
    timeout=30)

# Build command
cmd = [
    sys.executable, "-u", str(script_path),
    "--epochs", EPOCHS,
    "--batch_size", BATCH_SIZE,
]
if HF_TOKEN:
    cmd.extend(["--hf_token", HF_TOKEN])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

print(f"Launching: {' '.join(cmd)}")
print(f"Log: {LOG}")

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid} log={LOG}")
