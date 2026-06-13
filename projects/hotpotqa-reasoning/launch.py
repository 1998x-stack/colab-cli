"""Colab bootstrap: pip install deps + spawn run.py detached.

Usage: colab exec -f launch.py --timeout 120
"""
import subprocess
import sys
import os


def main() -> None:
    print("[launch] Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "vllm", "datasets", "matplotlib", "-q",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ])

    print("[launch] Downloading HotpotQA data...")
    subprocess.check_call([sys.executable, "/content/load_data.py"])

    print("[launch] Spawning run.py...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    logfile = "/content/run.log"

    with open(logfile, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", "/content/run.py"],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"[launch] OK. PID={proc.pid} log={logfile}")
    print("[launch] Run check_progress.py to monitor.")


if __name__ == "__main__":
    main()
