"""Check ML tutorial progress on Colab VM. Run via: colab exec -f check_progress.py --timeout 30

Reports: process status, latest log lines, and output directory contents.
"""

import os, subprocess

LOGFILE = "/content/tutorial.log"
OUTPUT_DIR = "/content/tutorial-output"

# 1. Check if tutorial process is alive
result = subprocess.run(
    ["pgrep", "-f", "tutorial.py"], capture_output=True, text=True,
)
if result.stdout.strip():
    print(f"Status: RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Status: NOT RUNNING")

# 2. Latest log lines
print(f"\n── Last 15 log lines ({LOGFILE}) ──")
if os.path.exists(LOGFILE):
    lines = open(LOGFILE).readlines()
    for line in lines[-15:]:
        print(line.rstrip())
    print(f"\n(total: {len(lines)} lines)")
else:
    print("(no log file yet)")

# 3. Output directory
print(f"\n── Output ({OUTPUT_DIR}) ──")
if os.path.exists(OUTPUT_DIR):
    for root, dirs, files in os.walk(OUTPUT_DIR):
        rel = os.path.relpath(root, OUTPUT_DIR)
        depth = len(rel.split(os.sep)) if rel != "." else 0
        prefix = "  " * depth
        name = os.path.basename(root) if depth > 0 else OUTPUT_DIR
        print(f"{prefix}{name}/")
        for fn in sorted(files):
            path = os.path.join(root, fn)
            size_kb = os.path.getsize(path) / 1024
            print(f"{prefix}  {fn} ({size_kb:.0f} KB)")
else:
    print("(no output yet)")

# 4. Summary JSON quick peek
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
if os.path.exists(summary_path):
    import json
    s = json.load(open(summary_path))
    overall = s.get("overall", {})
    print(f"\n── Summary ──")
    print(f"  NLP accuracy:    {overall.get('nlp_accuracy', '?'):.4f}" if isinstance(overall.get('nlp_accuracy'), float) else f"  NLP:    pending")
    print(f"  CV accuracy:     {overall.get('cv_accuracy', '?'):.4f}" if isinstance(overall.get('cv_accuracy'), float) else f"  CV:     pending")
    print(f"  Audio accuracy:  {overall.get('audio_accuracy', '?'):.4f}" if isinstance(overall.get('audio_accuracy'), float) else f"  Audio:  pending")
    if overall.get("average_accuracy"):
        print(f"  Average:         {overall['average_accuracy']:.4f}")
    if overall.get("total_train_time_minutes"):
        print(f"  Total time:      {overall['total_train_time_minutes']}m")
