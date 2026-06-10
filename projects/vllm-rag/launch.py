"""Launch vLLM RAG pipeline on Colab VM.

Spawns bootstrap.py as a detached subprocess that handles
pip install + server start + eval run. Returns immediately.
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
