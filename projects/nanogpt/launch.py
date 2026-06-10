"""Launch nanoGPT training on Colab VM via detached subprocess."""
import subprocess, sys, os

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "numpy", "requests", "matplotlib", "-q"])

# Upload training script is already at /content/train_nanogpt.py
# Start training in background, fully detached
logfile = "/content/nanogpt_train.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train_nanogpt.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid} log={logfile}")
