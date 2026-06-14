"""Launch dl-training benchmarks as detached subprocess on Colab VM.

Reads EXP_IDS env var (comma-separated) to select which experiments to run.
Default: all planned experiments.
"""
import subprocess, sys, os

# Resolve exp_ids from env or use default set
exp_ids = os.environ.get(
    "EXP_IDS",
    "dltrain-003,dltrain-004,dltrain-005,dltrain-006,dltrain-007,dltrain-008,dltrain-009,dltrain-010,dltrain-012"
)

# Install deps (matplotlib for optional plots)
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

log_path = "/content/dl-train-launch.log"
with open(log_path, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py", "--exp_ids", exp_ids],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
print(f"OK. PID={proc.pid}")
print(f"   exp_ids={exp_ids}")
print(f"   log={log_path}")
print(f"   outputs=/content/dl-training-output/")
