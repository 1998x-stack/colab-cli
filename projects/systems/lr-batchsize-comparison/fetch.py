"""Tar all experiment outputs on VM for cron download.

Run via: colab exec -f fetch.py --timeout 15
Output: /content/lr-bs-output.tar.gz
"""
import glob
import json
import os
import tarfile

OUT_DIR = "/content/lr-bs-output"
TAR_PATH = "/content/lr-bs-output.tar.gz"

# Report what we have
summary = {"experiments": {}}
for exp_dir in sorted(glob.glob(f"{OUT_DIR}/bs*_lr*")):
    name = os.path.basename(exp_dir)
    csv_path = f"{exp_dir}/metrics.csv"
    log_path = f"{exp_dir}/logs/train.log"
    summary_path = f"{exp_dir}/summary.json"

    n_lines = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            n_lines = len(f.readlines()) - 1  # exclude header

    exp_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            exp_summary = json.load(f)

    summary["experiments"][name] = {
        "csv_rows": n_lines,
        "summary": exp_summary,
    }
    print(f"[fetch] {name}: {n_lines} eval rows, summary={exp_summary.get('best_acc', '?')}")

# Also include launch log
launch_logs = glob.glob("/content/launch_bs*.log")
for ll in launch_logs:
    print(f"[fetch] launch log: {ll}")

print(f"[fetch] Total experiments found: {len(summary['experiments'])}")

# Tar everything
with tarfile.open(TAR_PATH, "w:gz") as tar:
    if os.path.exists(OUT_DIR):
        tar.add(OUT_DIR, arcname="lr-bs-output")
    for ll in launch_logs:
        tar.add(ll, arcname=os.path.basename(ll))

size_mb = os.path.getsize(TAR_PATH) / (1024 * 1024)
print(f"[fetch] Created {TAR_PATH} ({size_mb:.1f} MB)")
