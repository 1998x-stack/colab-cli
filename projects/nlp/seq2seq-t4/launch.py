"""Colab launcher: pip install deps, spawn training detached.

Usage:
    colab exec -f launch.py --timeout 120
"""
import subprocess
import sys
import os
import time

# --- Config ---
SCRIPT = "train.py"
DEPS = ["torch", "tokenizers", "sacrebleu", "matplotlib", "numpy"]
LOG = "/content/seq2seq-t4/logs/train.log"


def install_deps():
    for pkg in DEPS:
        print(f"[launch] pip install {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True,
        )


def upload_source_files():
    """Upload train.py, model.py, dataset.py to /content/ via colab exec.

    These need to already be on the VM for the detached subprocess to reference.
    Since colab exec -f sends the current file, we upload dependencies here.
    """
    # Files are already uploaded via: colab upload model.py /content/model.py etc.
    # This is just a reminder — the user should upload before launch.
    required = ["/content/train.py", "/content/model.py", "/content/dataset.py"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"[launch] WARNING: {missing} not found on VM.")
        print("[launch] Upload them first:")
        print("  colab upload train.py /content/train.py")
        print("  colab upload model.py /content/model.py")
        print("  colab upload dataset.py /content/dataset.py")
        return False
    return True


def spawn_training():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    script_path = f"/content/{SCRIPT}"
    print(f"[launch] Starting {script_path} ...")

    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path,
             "--epochs", "10", "--batch_size", "64", "--reverse_src"],
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
    if upload_source_files():
        spawn_training()
    else:
        print("[launch] Cannot proceed — upload missing files first.")
