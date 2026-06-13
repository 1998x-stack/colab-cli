"""Template launcher: pip install deps, spawn benchmark detached.

Usage:
    1. Customize SCRIPT and DEPS below
    2. Launch:  colab exec -f launch.py --timeout 120
"""

import subprocess
import sys
import os
import time

# ---------------------------------------------------------------------------
SCRIPT = "train.py"
DEPS = ["matplotlib"]
LOG = "/content/train.log"
# ---------------------------------------------------------------------------


def install_deps():
    for pkg in DEPS:
        print(f"[launch] pip install {pkg} ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True,
        )


def spawn_training():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    script_path = f"/content/{SCRIPT}"
    print(f"[launch] Starting {script_path} ...", flush=True)

    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(3)
    if proc.poll() is not None:
        print(f"[launch] ERROR: script exited immediately (code={proc.returncode}).", flush=True)
        print("[launch] Log tail:", flush=True)
        subprocess.run(["tail", "-20", LOG])
        sys.exit(1)

    print(f"[launch] OK. PID={proc.pid}  log={LOG}", flush=True)


if __name__ == "__main__":
    install_deps()
    spawn_training()
