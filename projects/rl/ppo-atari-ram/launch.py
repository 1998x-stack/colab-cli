#!/usr/bin/env python3
"""Launch PPO Atari RAM training as detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium[atari]", "ale-py", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/ppo-atari-output/logs/train.log"

print("=== Colab PPO Atari RAM Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/ppo-atari-output/logs", exist_ok=True)
os.makedirs("/content/ppo-atari-output/pngs", exist_ok=True)
os.makedirs("/content/ppo-atari-output/checkpoints", exist_ok=True)

print(f"\nLaunching {SCRIPT} detached...")
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
print("Output dir: /content/ppo-atari-output/")

time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
