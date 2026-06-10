"""Template launcher: check proxy, install deps, spawn training.

If clash proxy is running on 127.0.0.1:7890, routes traffic through it.
If not, uses direct connection (Colab VMs have good direct internet).

Usage:
    1. Customize SCRIPT and DEPS below
    2. Optional: bootstrap proxy first:  colab exec -f vm-proxy-bootstrap.py
    3. Launch:                           colab exec -f launch_proxy.py --timeout 120
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

MIXED_PORT = 7890


def proxy_running():
    """Check if clash proxy is reachable."""
    result = subprocess.run(
        ["curl", "-s", "--max-time", "2",
         "-x", f"http://127.0.0.1:{MIXED_PORT}",
         "https://www.google.com", "-o", "/dev/null", "-w", "%{http_code}"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "200"


def install_deps():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if proxy_running():
        print("[launch] Proxy detected — routing pip through 127.0.0.1:7890")
        env.update({
            "HTTPS_PROXY": f"http://127.0.0.1:{MIXED_PORT}",
            "HTTP_PROXY": f"http://127.0.0.1:{MIXED_PORT}",
            "ALL_PROXY": f"socks5://127.0.0.1:{MIXED_PORT}",
        })
    else:
        print("[launch] No proxy — using direct connection")

    for pkg in DEPS:
        print(f"[launch] pip install {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            env=env, check=True,
        )


def spawn_training():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if proxy_running():
        env.update({
            "HTTPS_PROXY": f"http://127.0.0.1:{MIXED_PORT}",
            "HTTP_PROXY": f"http://127.0.0.1:{MIXED_PORT}",
            "ALL_PROXY": f"socks5://127.0.0.1:{MIXED_PORT}",
        })

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
