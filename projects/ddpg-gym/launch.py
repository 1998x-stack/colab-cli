#!/usr/bin/env python3
"""Launch DDPG training as a detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/ddpg-output/train.log"

print("=== Colab DDPG Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

# Ensure output dir exists on VM (train.py also creates it, but create early for log)
os.makedirs("/content/ddpg-output", exist_ok=True)

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
print(f"Check: tail -f {LOG}")
print("Output dir: /content/ddpg-output/")

# Wait a moment to confirm the process is still alive
time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
