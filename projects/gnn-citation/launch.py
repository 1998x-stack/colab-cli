#!/usr/bin/env python3
"""Launch GCN citation network training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["torch-geometric", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/gnn-citation-output/launch.log"

print("=== GCN Citation Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    print(f"  pip install {dep} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

# Verify imports
print("  verifying torch_geometric ...")
subprocess.check_call(
    [sys.executable, "-c", "import torch_geometric; print(f'  PyG {torch_geometric.__version__} OK')"])

os.makedirs("/content/gnn-citation-output", exist_ok=True)

print(f"\nLaunching {SCRIPT} detached ...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
print("Output root: /content/gnn-citation-output/")

time.sleep(5)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log:")
    try:
        with open(LOG) as f:
            print(f.read()[-2000:])
    except Exception:
        pass
