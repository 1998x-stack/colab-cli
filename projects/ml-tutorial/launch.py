"""Launch the ML tutorial on a Colab VM. Run via: colab exec -f launch.py --timeout 300

Installs dependencies, then spawns tutorial.py as a detached subprocess
so it survives after colab exec returns.
"""

import subprocess
import sys
import os

DEPS = [
    "transformers", "evaluate", "accelerate",
    "scikit-learn", "seaborn", "matplotlib",
    "torch", "torchvision",
]

# Install dependencies
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS
)

# Spawn training detached
logfile = "/content/tutorial.log"
print(f"[launch] Starting tutorial, log={logfile}")

with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/tutorial.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid}")
