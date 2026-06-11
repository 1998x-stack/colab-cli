"""Template launcher: pip install deps, spawn training detached.

Colab VMs have excellent direct internet from GCP (Google, PyPI, GitHub,
HuggingFace all reachable). VM-side proxy (SS servers) is virtually never
reachable from GCP, so this template skips proxy detection and uses direct
connections. If you specifically need VM-side proxy, use vm-proxy-bootstrap.py
first, then adapt this template.

Usage:
    1. Customize SCRIPT and DEPS below
    2. Launch:  colab exec -f launch_proxy.py --timeout 120
"""

import subprocess
import sys
import os
import time

# ---------------------------------------------------------------------------
SCRIPT = "train.py"                     # Script to run on VM
DEPS = ["torch", "transformers"]        # pip packages to install
LOG = "/content/train.log"              # Log output path
# ---------------------------------------------------------------------------


def install_deps():
    for pkg in DEPS:
        print(f"[launch] pip install {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True,
        )


def spawn_training():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    script_path = f"/content/{SCRIPT}"
    print(f"[launch] Starting {script_path} ...")

    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(3)
    if proc.poll() is not None:
        print(f"[launch] ERROR: script exited immediately (code={proc.returncode}).")
        print("[launch] Log tail:")
        subprocess.run(["tail", "-20", LOG])
        sys.exit(1)

    print(f"[launch] OK. PID={proc.pid}  log={LOG}")


if __name__ == "__main__":
    install_deps()
    spawn_training()
