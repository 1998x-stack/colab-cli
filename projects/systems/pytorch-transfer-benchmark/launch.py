"""Launch benchmark_transfer.py as detached subprocess on Colab VM."""
import subprocess
import sys
import os

# Install matplotlib (only extra dep beyond torch)
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open("/content/benchmark_launch.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/benchmark_transfer.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
print(f"OK. PID={proc.pid}")
print(f"   log=/content/benchmark_launch.log")
print(f"   outputs=/content/pytorch-transfer-benchmark-output/")
