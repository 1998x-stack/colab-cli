#!/usr/bin/env python3
"""Launch RAG FastText training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["fasttext", "faiss-cpu>=1.9.0", "scipy", "numpy", "datasets", "matplotlib", "scikit-learn"]
SCRIPT = "train.py"
LOG = "/content/rag-fasttext-output/logs/train.log"

print("=== Colab RAG: FastText + BM25 + FAISS Launcher ===")
print(f"Installing: {DEPS}")
for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/rag-fasttext-output/logs", exist_ok=True)
os.makedirs("/content/rag-fasttext-output/pngs", exist_ok=True)
os.makedirs("/content/rag-fasttext-output/checkpoints", exist_ok=True)

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
