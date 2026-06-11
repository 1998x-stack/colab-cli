"""Launch a Python script as a detached background job on a Colab VM.

Customize SCRIPT and DEPS below, then run:
    colab upload launch.py launch.py
    colab exec -f launch.py --timeout 120

The script is spawned with start_new_session=True so it survives after
colab exec returns. Output goes to the log file with unbuffered stdout.
"""

import subprocess
import sys
import os

# ── Customize these ──────────────────────────────────────────────────────
SCRIPT = "/content/train.py"              # path to your training script on the VM
DEPS = ["torch"]                          # pip packages to install
LOG = "/content/train.log"               # where stdout/stderr goes
# ─────────────────────────────────────────────────────────────────────────

# Install dependencies
for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

# Start training in background, fully detached
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from kernel — survives exec timeout
        env=env,
    )

print(f"OK. PID={proc.pid} log={LOG}")
print("Check progress: colab exec -f check_progress.py --timeout 15")
