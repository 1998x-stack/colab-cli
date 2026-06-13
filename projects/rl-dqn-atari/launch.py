"""Launch DQN Atari training on Colab VM. Run via: colab exec -f launch.py --timeout 300

Installs system deps (libcairo2 for Atari rendering), Python deps,
then spawns train.py as a detached subprocess.
"""

import subprocess
import sys
import os

# System dependencies for Atari / ALE rendering
try:
    subprocess.run(
        ["apt-get", "update", "-qq"], capture_output=True, timeout=60,
    )
    subprocess.check_call(
        ["apt-get", "install", "-y", "-qq", "libcairo2-dev", "libpango1.0-dev"],
        timeout=120,
    )
    print("[launch] System deps installed")
except Exception as e:
    print(f"[launch] System deps skipped (may already be installed): {e}")

# Python dependencies
DEPS = ["gymnasium[atari]", "ale-py", "opencv-python-headless", "torch", "matplotlib"]
print("[launch] Installing Python dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS
)

# Spawn training detached
logfile = "/content/dqn_train.log"
print(f"[launch] Starting DQN training, log={logfile}")

with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid}")
