"""Launch script for Colab — pip install deps and spawn train.py as detached subprocess.

Config via env vars (all optional):
  LAUNCH_SCRIPT: training script path (default: train.py)
  LAUNCH_DEPS: comma-separated pip packages (default: torch,torchvision,datasets,matplotlib,seaborn,scikit-learn)
  LAUNCH_LOG: log file path (default: /content/cnn-explainer-output/logs/launch.log)
  LAUNCH_ARGS: extra args passed to train.py (default: empty)
"""

import os
import subprocess
import sys

SCRIPT = os.environ.get("LAUNCH_SCRIPT", "train.py")
DEPS = os.environ.get("LAUNCH_DEPS", "torch,torchvision,datasets,matplotlib,seaborn,scikit-learn")
LOG = os.environ.get("LAUNCH_LOG", "/content/cnn-explainer-output/logs/launch.log")
ARGS = os.environ.get("LAUNCH_ARGS", "")

os.makedirs(os.path.dirname(LOG), exist_ok=True)

deps_list = [d.strip() for d in DEPS.split(",") if d.strip()]
print(f"[launch] Installing: {deps_list}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + deps_list)

print(f"[launch] Spawning: {sys.executable} -u {SCRIPT} {ARGS}  (log → {LOG})")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT] + (ARGS.split() if ARGS else []),
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid}  log={LOG}")
