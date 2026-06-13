"""Check autoresearch training progress on Colab VM."""

import os
import subprocess
import json

LOGFILE = "/content/train.log"
OUTPUT_DIR = "/content/autoresearch-output"

# 1. Process status
result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Status: RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Status: NOT RUNNING (may have completed)")

# 2. Latest log
print(f"\n── Last 15 lines ({LOGFILE}) ──")
if os.path.exists(LOGFILE):
    lines = open(LOGFILE).readlines()
    for line in lines[-15:]:
        print(line.rstrip())
    print(f"\n(total: {len(lines)} lines)")
else:
    print("(no log yet)")

# 3. Training progress
for l in reversed(lines) if os.path.exists(LOGFILE) else []:
    if "step" in l and "%" in l:
        print(f"\nLatest: {l.strip()}")
        break

# 4. Output
print(f"\n── Output ({OUTPUT_DIR}) ──")
if os.path.exists(OUTPUT_DIR):
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, fn)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fn} ({size_kb:.0f} KB)")
        if fn == "metrics.json":
            m = json.load(open(path))
            print(f"    val_bpb: {m.get('val_bpb', '?')}")
            print(f"    params: {m.get('num_params', '?'):,}")
            print(f"    tokens: {m.get('total_tokens', '?'):,}")
else:
    print("  (no output yet)")
