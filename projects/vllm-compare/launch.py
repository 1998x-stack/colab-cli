"""Launch vLLM comparison benchmark on Colab VM.

Spawns bootstrap.py as a detached subprocess that handles
pip install + benchmark execution. Returns immediately so
the colab exec WebSocket can disconnect without issues.
"""
import subprocess, sys, os

print("[launch] Spawning bootstrap...")
logfile = "/content/bootstrap.log"
with open(logfile, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/bootstrap.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(f"[launch] OK. Bootstrap PID={proc.pid} log={logfile}")
print("[launch] Run check_progress.py to monitor.")
