"""Fire-and-forget bootstrap: spawn launch.py detached, return immediately."""
import subprocess
import sys
import os

logfile = "/content/run.log"
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(logfile, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/launch.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
print(f"OK. Bootstrap PID={proc.pid} log={logfile}")
