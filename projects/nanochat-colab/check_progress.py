"""Check nanochat training progress on Colab VM.

Run via: cb exec -f check_progress.py --timeout 15
"""

import os
import subprocess

LOG_FILE = "/content/train.log"
BASE_DIR = "/content/nanochat-data"

# 1. Process status
print("── Process Status ──")
result = subprocess.run(["pgrep", "-f", "base_train"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Status: RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Status: NOT RUNNING (may have completed)")

# 2. Latest log tail
print(f"\n── Last 20 lines ({LOG_FILE}) ──")
if os.path.exists(LOG_FILE):
    lines = open(LOG_FILE).readlines()
    for line in lines[-20:]:
        print(line.rstrip())
    print(f"\n(total: {len(lines)} lines)")

    # Extract latest metrics
    for l in reversed(lines):
        if "step " in l and "%" in l and "loss:" in l:
            print(f"\nLatest step: {l.strip()}")
            break
    for l in reversed(lines):
        if "Validation bpb" in l:
            print(f"Latest val:   {l.strip()}")
            break
    for l in reversed(lines):
        if "Minimum validation bpb" in l:
            print(f"*** {l.strip()} ***")
            break
else:
    print("(no log yet)")

# 3. Checkpoint files
ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", "d6")
print(f"\n── Checkpoints ({ckpt_dir}) ──")
if os.path.exists(ckpt_dir):
    for fn in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, fn)
        if os.path.isdir(path):
            size_kb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, filenames in os.walk(path)
                for f in filenames
            ) / 1024
            print(f"  {fn}/ ({size_kb:.0f} KB)")
        else:
            print(f"  {fn} ({os.path.getsize(path)/1024:.0f} KB)")
else:
    print("  (no checkpoints yet)")

# 4. Dataset info
print(f"\n── Data ({BASE_DIR}) ──")
for subdir in ["base_data_climbmix", "tokenizer"]:
    path = os.path.join(BASE_DIR, subdir)
    exists = os.path.exists(path)
    if exists:
        items = len(os.listdir(path))
        print(f"  {subdir}/ exists ({items} items)")
    else:
        print(f"  {subdir}/ (not found)")
