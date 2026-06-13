"""Colab bootstrap: pip install matplotlib+pandas, spawn train.py as nohup subprocess.

Reads /content/exp_id.txt for experiment name (used in log tags).
"""
import os
import subprocess
import sys

EXP_ID_PATH = "/content/exp_id.txt"
LOG = "/content/train.log"
OUTPUT_DIR = "/content/transformer-kv-cache-output"


def main():
    exp_id = "default"
    if os.path.exists(EXP_ID_PATH):
        with open(EXP_ID_PATH) as f:
            exp_id = f.read().strip()
    print(f"[launch] Exp ID: {exp_id}")

    print("[launch] Installing deps ...")
    for dep in ["matplotlib", "pandas"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = (
        f"{sys.executable} -u /content/train.py "
        f"--output_dir {OUTPUT_DIR} "
        f"--device cuda --max_epochs 10 "
        f"--batch_size 128 --block_size 128 --text_limit 50000 "
    )

    print(f"[launch] Running: {cmd}")
    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            cmd.split(), stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
    print(f"[launch] Train PID={proc.pid}, log={LOG}")
    print("[launch] DONE. Training running detached.")


if __name__ == "__main__":
    main()
