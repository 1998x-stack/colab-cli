#!/usr/bin/env python3
"""Launch FastText PyTorch training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["datasets", "matplotlib", "numpy"]
SCRIPT = "train.py"
LOG = "/content/fasttext-pytorch-output/logs/train.log"

print("=== FastText PyTorch Launcher ===")
print(f"Installing: {DEPS}")
for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/fasttext-pytorch-output/logs", exist_ok=True)
os.makedirs("/content/fasttext-pytorch-output/pngs", exist_ok=True)
os.makedirs("/content/fasttext-pytorch-output/checkpoints", exist_ok=True)

# Read HF token from uploaded file if present
hf_token_path = "/content/.huggingface_token"
if os.path.exists(hf_token_path):
    with open(hf_token_path) as f:
        token = f.read().strip()
    os.environ["HF_TOKEN"] = token
    print("HF_TOKEN set from /content/.huggingface_token")
elif os.path.exists(os.path.expanduser("~/.huggingface/token")):
    print("Using ~/.huggingface/token")
else:
    print("WARNING: No HF token found — public datasets only")

print(f"\nLaunching {SCRIPT} detached ...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
