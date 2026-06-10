"""Launch sac_mountaincar.py via nohup on Colab VM."""
import subprocess, sys, os

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium", "-q"])

# Start training in background, fully detached
logfile = "/content/sac_train.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/sac_mountaincar.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid} log={logfile}")
