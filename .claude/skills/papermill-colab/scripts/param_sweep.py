"""Template: hyperparameter sweep via papermill — runs notebook N times with
different parameter sets, each producing a distinct output notebook.

All runs share a single log file. Each output notebook is named after its
parameters so results don't collide.

Usage:
    1. Customize NOTEBOOK, PARAM_GRID, and DEPS below
    2. Upload notebook to VM:  colab upload notebook.ipynb /content/notebook.ipynb
    3. Launch:  colab exec -f param_sweep.py --timeout 30
    4. The launcher returns in <10s. The sweep runs detached.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
NOTEBOOK = "notebook.ipynb"          # Input notebook path on VM
PARAM_GRID = [                       # Each dict is one papermill run
    {"epochs": 30, "lr": 0.01},
    {"epochs": 30, "lr": 0.001},
    {"epochs": 50, "lr": 0.01},
    {"epochs": 50, "lr": 0.001},
]
DEPS = []                            # Extra pip packages
LOG = "/content/sweep.log"           # Log output path
# ---------------------------------------------------------------------------

PAPERMILL_DEPS = ["papermill", "ipykernel"]


def output_name(params):
    """Generate a unique output filename from parameters."""
    parts = []
    for k, v in sorted(params.items()):
        v_str = str(v).replace(".", "_").replace("-", "m")
        parts.append(f"{k}{v_str}")
    return f"output_{'_'.join(parts)}.ipynb"


def log(msg, f):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def install_deps(f):
    all_deps = PAPERMILL_DEPS + DEPS
    for pkg in all_deps:
        log(f"pip install {pkg} ...", f)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True,
        )


def run_sweep():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(LOG, "w") as f:
        install_deps(f)
        log(f"Sweep: {len(PARAM_GRID)} configs", f)

        for i, params in enumerate(PARAM_GRID):
            output = output_name(params)
            cmd = [sys.executable, "-m", "papermill", NOTEBOOK, output]
            for k, v in params.items():
                cmd.extend(["-p", str(k), str(v)])

            log(f"[{i+1}/{len(PARAM_GRID)}] {' '.join(cmd)}", f)
            t0 = time.time()

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAIL (code={result.returncode})"
            log(f"[{i+1}/{len(PARAM_GRID)}] {status} | {elapsed:.0f}s | -> {output}", f)

        log("Sweep complete.", f)


if __name__ == "__main__":
    run_sweep()
