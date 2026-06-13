#!/usr/bin/env python3
"""Launch DDQN vs NoisyNet training as a detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/ddqn-noisy-output/train.log"

print("=== Colab DDQN vs NoisyNet Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/ddqn-noisy-output", exist_ok=True)

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
print("Output dir: /content/ddqn-noisy-output/")

time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
