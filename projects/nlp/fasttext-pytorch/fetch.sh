#!/bin/bash
# Fetch FastText training results from Colab VM (REST download, survives WebSocket drops).
# Called by cron every 4 minutes. Excludes checkpoints from download.
#
# Usage: ./fetch.sh [session_name]

set -euo pipefail

SESSION="${1:-fasttext-train}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
COLB_BIN=/Users/mx/.local/bin/colab

mkdir -p "$LOCAL_OUT"

# Step 1: Tar output on VM (excluding checkpoints)
echo "[fetch] $(date '+%H:%M:%S') Tarring output on VM..."
echo '
import subprocess as s
r = s.run([
    "tar", "-czf", "/content/fasttext-pytorch-output.tar.gz",
    "-C", "/content/fasttext-pytorch-output",
    "--exclude=checkpoints",
    "."
], capture_output=True)
if r.returncode != 0:
    print(f"tar stderr: {r.stderr.decode()}")
' | "$COLB_BIN" exec -s "$SESSION" --timeout 30 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed (WebSocket may be down)"
}

# Step 2: Download tar
echo "[fetch] Downloading..."
"$COLB_BIN" download -s "$SESSION" \
    /content/fasttext-pytorch-output.tar.gz \
    "$LOCAL_OUT/output.tar.gz" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or tar missing"
    exit 0
}

# Step 3: Extract
cd "$LOCAL_OUT"
tar -xzf output.tar.gz 2>/dev/null || {
    echo "[fetch] WARNING: extract failed"
    exit 0
}

# Step 4: Report
echo "[fetch] $(date '+%H:%M:%S') Done."

if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""
    echo "── Last 10 log lines ──"
    tail -10 "$LOCAL_OUT/logs/train.log"
fi

if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""
    echo "── Metrics CSV ──"
    head -1 "$LOCAL_OUT/metrics.csv"
    tail -5 "$LOCAL_OUT/metrics.csv"
fi

echo ""
echo "── PNGs ──"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"

if [ -f "$LOCAL_OUT/summary.json" ]; then
    echo ""
    echo "── Summary ──"
    python3 -c "
import json
with open('$LOCAL_OUT/summary.json') as f:
    s = json.load(f)
print(f\"  test_acc={s.get('test_acc', 'N/A')}  epochs={s.get('epochs_completed', 'N/A')}  time={s.get('total_time_s', 0)/60:.1f}m\")
" 2>/dev/null || echo "(summary parse failed)"
fi

echo ""
echo "Files in: $LOCAL_OUT"
