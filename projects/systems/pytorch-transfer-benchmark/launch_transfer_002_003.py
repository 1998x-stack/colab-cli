"""Launch transfer-002 and transfer-003 benchmarks as detached subprocesses on Colab VM."""
import subprocess
import sys
import os
import time

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

scripts = [
    "/content/benchmark_transfer_002.py",
    "/content/benchmark_transfer_003.py",
]

for script in scripts:
    name = script.replace("/content/benchmark_", "").replace(".py", "")
    log_file = f"/content/{name}.log"
    print(f"Launching {script} → {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"  PID={proc.pid}")
    time.sleep(2)  # small gap so they don't race on CUDA init

print("All benchmarks launched.")
print("Check progress: colab exec -f check_progress.py")
