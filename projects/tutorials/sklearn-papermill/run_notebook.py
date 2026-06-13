"""Launcher for sklearn tutorial notebook — pip install papermill + sklearn deps,
spawn papermill detached. Template customized from papermill-colab skill.

Usage:
    1. Upload notebook + utils to VM (see deploy.sh)
    2. Launch:  colab exec -f run_notebook.py --timeout 30
    3. The launcher returns in <10s. Papermill runs detached on the VM.
"""

import subprocess
import sys
import os
import time

# ---------------------------------------------------------------------------
NOTEBOOK = "tutorial.ipynb"                 # Input notebook on VM
OUTPUT = "/content/tutorial-output-nb.ipynb"  # Executed notebook output
PARAMS = {
    "n_epochs": 15,
    "alpha": 0.0001,
    "test_size": 0.2,
    "random_state": 42,
}
DEPS = ["numpy", "matplotlib", "scikit-learn"]
LOG = "/content/papermill.log"
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
    print(f"[launch] Output: {OUTPUT}")
    print(f"[launch] Params: {PARAMS}")


if __name__ == "__main__":
    install_deps()
    spawn_papermill()
