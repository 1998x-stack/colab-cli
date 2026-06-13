"""Template launcher: pip install papermill + deps, execute notebook detached.

Colab VMs have excellent direct internet from GCP. Papermill + dependencies
install quickly (papermill ~2MB, small transitive deps).

Usage:
    1. Customize NOTEBOOK, OUTPUT, PARAMS, and DEPS below
    2. Upload notebook to VM:  colab upload notebook.ipynb /content/notebook.ipynb
    3. Launch:  colab exec -f run_notebook.py --timeout 30
    4. The launcher returns in <10s. Papermill runs detached.
"""

import subprocess
import sys
import os
import time

# ---------------------------------------------------------------------------
NOTEBOOK = "notebook.ipynb"          # Input notebook path on VM
OUTPUT = "output.ipynb"              # Output notebook path on VM
PARAMS = {}                          # {"epochs": 50, "lr": 0.001} or {}
DEPS = []                            # Extra pip packages (e.g., ["numpy", "matplotlib"])
LOG = "/content/papermill.log"       # Log output path
# ---------------------------------------------------------------------------

PAPERMILL_DEPS = ["papermill", "ipykernel"]


def install_deps():
    all_deps = PAPERMILL_DEPS + DEPS
    for pkg in all_deps:
        print(f"[launch] pip install {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True,
        )


def build_papermill_cmd():
    cmd = [sys.executable, "-m", "papermill", NOTEBOOK, OUTPUT]
    for key, val in PARAMS.items():
        cmd.extend(["-p", str(key), str(val)])
    return cmd


def spawn_papermill():
    cmd = build_papermill_cmd()
    print(f"[launch] Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(3)
    if proc.poll() is not None:
        print(f"[launch] ERROR: papermill exited immediately (code={proc.returncode}).")
        print("[launch] Log tail:")
        subprocess.run(["tail", "-20", LOG])
        sys.exit(1)

    print(f"[launch] OK. PID={proc.pid}  log={LOG}")
    print(f"[launch] Input:  /content/{NOTEBOOK}")
    print(f"[launch] Output: /content/{OUTPUT}")
    if PARAMS:
        print(f"[launch] Params: {PARAMS}")


if __name__ == "__main__":
    install_deps()
    spawn_papermill()
