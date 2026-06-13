"""Launch SARSA training on Colab VM — install deps + spawn detached."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/rl-sarsa-output/logs/train.log"

# Ensure output dirs exist
os.makedirs("/content/rl-sarsa-output/logs", exist_ok=True)
os.makedirs("/content/rl-sarsa-output/pngs", exist_ok=True)
os.makedirs("/content/rl-sarsa-output/checkpoints", exist_ok=True)

# Install deps
for pkg in DEPS:
    print(f"[launch] pip install {pkg} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

# Verify gymnasium works
subprocess.run([sys.executable, "-c", "import gymnasium; print(f'gymnasium {gymnasium.__version__} OK')"], check=True)

# Spawn training detached
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

script_path = f"/content/{SCRIPT}"
print(f"[launch] Starting {script_path} ...")

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

time.sleep(3)
if proc.poll() is not None:
    print(f"[launch] ERROR: script exited immediately (code={proc.returncode}).")
    print("[launch] Log tail:")
    subprocess.run(["tail", "-20", LOG])
    sys.exit(1)

print(f"[launch] OK. PID={proc.pid}  log={LOG}")
print(f"[launch] Check: tail -f {LOG}")
print("[launch] Output: /content/rl-sarsa-output/")
