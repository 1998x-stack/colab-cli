"""Relay watchdog — keeps WebSocket alive while training runs in background.

Upload once: colab upload watchdog.py /content/text2sql_finetune/watchdog.py
Then: colab exec -s <name> -f watchdog.py --timeout 420

Auto-names itself (ws-1, ws-2, ...) from a counter file on the VM.
For short training (3-4 min, this project's default), a single bg_launch.py
suffices. Use this watchdog for hyperparameter sweeps or larger datasets
where training exceeds 8 minutes.

Protocol (from live tests, 2026-06-14):
  - 5-min watchdog window (safe inside China WS stability)
  - 25s heartbeat interval (nvidia-smi + log reads → real TCP payload)
  - Launch next watchdog 30s before current exits
  - For redundancy: launch 2 watchdogs per handoff window
"""
import subprocess, os, time
from datetime import datetime, timezone

OUTPUT_DIR = "/content/text2sql-finetune-output"
PROJECT_DIR = "/content/text2sql_finetune"
LOG_DIR = f"{PROJECT_DIR}/logs"
COUNTER_FILE = f"{OUTPUT_DIR}/watchdog_counter"
TRAIN_PID_FILE = f"{OUTPUT_DIR}/train.pid"
WD_LOG = f"{OUTPUT_DIR}/logs/watchdog.log"

DURATION = 300     # 5 minutes
INTERVAL = 25      # heartbeat every 25s

os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

# Auto-increment watchdog name
counter = 0
if os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE) as f:
        counter = int(f.read().strip()) + 1
with open(COUNTER_FILE, "w") as f:
    f.write(str(counter))
NAME = f"ws-{counter}"

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(WD_LOG, "a") as f:
        f.write(line + "\n")

wlog(f"START pid={os.getpid()} window={DURATION}s")

# ── Find training ────────────────────────────────────────
train_pid = None
if os.path.exists(TRAIN_PID_FILE):
    with open(TRAIN_PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"Training PID={train_pid} ALIVE")
    except OSError:
        wlog(f"Training PID={train_pid} DEAD")
        train_pid = None
else:
    # Check ps as fallback
    try:
        result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            train_pid = int(result.stdout.strip().split("\n")[0])
            wlog(f"Training found via pgrep: PID={train_pid}")
    except Exception:
        pass

if train_pid is None:
    wlog("WARNING: no training process found — monitor-only mode")

# ── GPU check ────────────────────────────────────────────
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    wlog(f"GPU={gpu}")
except Exception:
    wlog("GPU check failed")

# ── Check for completion ─────────────────────────────────
adapter_path = f"{PROJECT_DIR}/lora_weights/adapter_config.json"
eval_report = f"{OUTPUT_DIR}/eval_report.json"

if os.path.exists(eval_report):
    wlog("Eval report exists — training + eval already complete")
elif os.path.exists(adapter_path):
    wlog("LoRA weights exist — training complete, eval may be pending")

# ── Watchdog loop ────────────────────────────────────────
start = time.time()
iteration = 0

while time.time() - start < DURATION:
    iteration += 1
    elapsed = time.time() - start

    # Training status
    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"ALIVE(PID={train_pid})"
        except OSError:
            train_status = "DEAD"
            wlog(f"Training process exited at t={elapsed:.0f}s")

    # LoRA weights
    lora_status = "ready" if os.path.exists(adapter_path) else "training"

    # Eval
    eval_status = "done" if os.path.exists(eval_report) else "pending"

    # GPU
    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True, text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    # Training progress (last log line)
    log_tail = "(no log)"
    train_log = f"{LOG_DIR}/train.log"
    if os.path.exists(train_log):
        try:
            with open(train_log) as f:
                lines = f.readlines()
                log_tail = lines[-1].strip()[-200:] if lines else "(empty)"
        except Exception:
            pass

    wlog(f"iter={iteration} elapsed={elapsed:.0f}s "
         f"train={train_status} lora={lora_status} eval={eval_status} "
         f"gpu=[{gpu_info}] tail: {log_tail}")

    # Heartbeat to stdout — real TCP payload for NAT timeout reset
    print(f"[{ts()}] {NAME} heartbeat "
          f"iter={iteration} elapsed={elapsed:.0f}s train={train_status}",
          flush=True)

    if train_status == "DEAD" and os.path.exists(adapter_path):
        wlog("Training complete (weights saved) — continuing to hold session")
    elif train_status == "DEAD" and not os.path.exists(adapter_path):
        wlog("Training died before completion — check train.log")

    time.sleep(INTERVAL)

total = time.time() - start
wlog(f"EXIT total={total:.0f}s iterations={iteration}")
wlog(f"HANDOFF: launch ws-{counter+1} with: colab exec -f watchdog.py --timeout 420")
