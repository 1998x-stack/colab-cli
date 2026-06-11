"""Colab bootstrap: pip install deps, spawn train.py as detached subprocess.

Reads /content/exp_id.txt for experiment config.
Supports --resume flag via /content/resume_path.txt if checkpoint exists.
"""
import subprocess, sys, os

EXP_ID_PATH = "/content/exp_id.txt"
RESUME_PATH_FILE = "/content/resume_path.txt"
LOG = "/content/train.log"
DEPS = ["tokenizers", "sacrebleu", "matplotlib", "datasets"]

# --- Read experiment ID ---
with open(EXP_ID_PATH) as f:
    exp_id = f.read().strip()
print(f"[launch] Exp ID: {exp_id}")

# --- Check for resume checkpoint ---
resume_flag = ""
if os.path.exists(RESUME_PATH_FILE):
    with open(RESUME_PATH_FILE) as f:
        ckpt_path = f.read().strip()
    if ckpt_path and os.path.exists(ckpt_path):
        resume_flag = f"--resume {ckpt_path}"
        resume_epoch = os.path.basename(ckpt_path).replace("checkpoint_epoch", "").replace(".pt", "")
        print(f"[launch] Resuming from epoch {resume_epoch}: {ckpt_path}")
    else:
        print(f"[launch] Resume path file exists but checkpoint not found at '{ckpt_path}' — starting fresh")
else:
    print("[launch] No resume checkpoint — starting fresh")

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Spawn training ---
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

cmd = f"{sys.executable} -u /content/train.py --exp_id {exp_id}"
if resume_flag:
    cmd += f" {resume_flag}"

print(f"[launch] Running: {cmd}")
with open(LOG, "w") as f:
    proc = subprocess.Popen(
        cmd.split(),
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[launch] Train PID={proc.pid}, log={LOG}")
print(f"[launch] DONE. Training running detached.")
