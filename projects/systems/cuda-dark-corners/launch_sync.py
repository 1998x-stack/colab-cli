"""Launch hidden-synchronization benchmarks (sync-001, sync-002, sync-003) as detached subprocesses."""
import subprocess
import sys
import os
import time

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

scripts = [
    "/content/benchmark_sync_001.py",
    "/content/benchmark_sync_002.py",
    "/content/benchmark_sync_003.py",
]

for script in scripts:
    name = script.replace("/content/benchmark_", "").replace(".py", "")
    log_file = f"/content/{name}.log"
    print(f"Launching {script} -> {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"  PID={proc.pid}")
    time.sleep(1)

print("All sync benchmarks launched.")
