import subprocess
import sys
import os

subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "numpy", "scipy", "-q"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open("/content/cuda_tutorial.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/cuda_tutorial.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid} log=/content/cuda_tutorial.log")
