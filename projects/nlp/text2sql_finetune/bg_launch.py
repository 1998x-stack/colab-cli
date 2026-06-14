"""Detached launcher: pip install + dataset prep + spawn training in background.

Upload source files, then:
    colab exec -s <name> -f bg_launch.py --timeout 120

Exits in <30s. Training spawns as detached subprocess (start_new_session=True) —
survives all WebSocket drops. Writes train.pid for watchdog monitoring.

For long training (>8 min), chain with watchdog.py relay:
    colab exec -s <name> -f watchdog.py --timeout 420
"""
import os, subprocess, sys, shutil, time

PROJECT_DIR = "/content/text2sql_finetune"
OUTPUT_DIR = "/content/text2sql-finetune-output"
DATA_DIR = f"{PROJECT_DIR}/data"
LOG_DIR = f"{PROJECT_DIR}/logs"

# Configurable via env vars
TRAIN_EXAMPLES = int(os.environ.get("T2S_TRAIN_EXAMPLES", "500"))
TEST_EXAMPLES = int(os.environ.get("T2S_TEST_EXAMPLES", "100"))
BATCH_SIZE = int(os.environ.get("T2S_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("T2S_GRAD_ACCUM", "2"))
MAX_STEPS = int(os.environ.get("T2S_MAX_STEPS", "0"))  # 0 = auto (all data)
LR = os.environ.get("T2S_LR", "2e-4")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

t0 = time.time()

# ── Install deps ─────────────────────────────────────────
print("[bg] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "peft", "datasets", "accelerate", "torchao"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print(f"[bg] Deps installed ({time.time() - t0:.0f}s)")

# ── Clear stale HF cache ─────────────────────────────────
hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)

# ── Prepare dataset ──────────────────────────────────────
print("[bg] Preparing dataset with auto-split...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "auto",
    "--train_examples", str(TRAIN_EXAMPLES),
    "--test_examples", str(TEST_EXAMPLES),
    "--train_output", f"{DATA_DIR}/train.pt",
    "--test_output", f"{DATA_DIR}/test.pt",
])
print(f"[bg] Dataset prepared ({time.time() - t0:.0f}s)")

# ── Spawn training as detached subprocess ────────────────
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open(f"{LOG_DIR}/train.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"{PROJECT_DIR}/train.py",
         "--data_path", f"{DATA_DIR}/train.pt",
         "--output_dir", f"{PROJECT_DIR}/lora_weights",
         "--max_steps", str(MAX_STEPS),
         "--batch_size", str(BATCH_SIZE),
         "--grad_accum", str(GRAD_ACCUM),
         "--lr", LR],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )

# Write PID for watchdog monitoring
with open(f"{OUTPUT_DIR}/train.pid", "w") as f:
    f.write(str(proc.pid))

# Initialize watchdog counter
with open(f"{OUTPUT_DIR}/watchdog_counter", "w") as f:
    f.write("0")

print(f"[bg] Training PID={proc.pid} — detached, survives WebSocket drops")
print(f"[bg] Monitor: tail /content/text2sql_finetune/logs/train.log")
print(f"[bg] PID file: {OUTPUT_DIR}/train.pid")
print(f"[bg] Watchdog: colab exec -s <name> -f /content/text2sql_finetune/watchdog.py --timeout 420")
print(f"[bg] Launcher done ({time.time() - t0:.0f}s)")
