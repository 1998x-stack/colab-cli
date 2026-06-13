#!/bin/bash
# Cron-friendly fetch: tar on VM → download → extract → report.
# Usage: SESSION=training bash fetch.sh
# Set COLAB_BIN to override path (for multi-account: COLAB_BIN="$HOME/colab-accounts/account-b/.local/bin/colab")

set -euo pipefail

SESSION="${SESSION:-training}"
COLAB_BIN="${COLAB_BIN:-/Users/mx/.local/bin/colab}"
OUTPUT_DIR="projects/cnn-explainer/output"
REMOTE_DIR="/content/cnn-explainer-output"
TAR_NAME="cnn-explainer-output.tar.gz"

mkdir -p "$OUTPUT_DIR"

# 1. Tar on VM
echo "[fetch] Taring $REMOTE_DIR on VM..."
echo "import subprocess, os; subprocess.run(['tar', '-czf', '/content/${TAR_NAME}', '-C', '/content', 'cnn-explainer-output'], check=True); print('OK')" | "$COLAB_BIN" exec -s "$SESSION" --timeout 30 2>&1 || echo "[fetch] WARNING: tar exec failed (may be OK if no new files)"

# 2. Download tar
echo "[fetch] Downloading..."
"$COLAB_BIN" download -s "$SESSION" "/content/${TAR_NAME}" "${OUTPUT_DIR}/${TAR_NAME}" 2>&1

# 3. Extract
echo "[fetch] Extracting..."
tar -xzf "${OUTPUT_DIR}/${TAR_NAME}" -C "$OUTPUT_DIR" 2>&1 || echo "[fetch] WARNING: extract failed"

# 4. Report
echo ""
echo "=== Metrics CSV (last 5 rows) ==="
CSV_FILE=$(find "$OUTPUT_DIR" -name "metrics.csv" -maxdepth 2 2>/dev/null | head -1)
if [ -n "$CSV_FILE" ] && [ -s "$CSV_FILE" ]; then
    tail -5 "$CSV_FILE"
else
    echo "  (no metrics yet)"
fi

echo ""
echo "=== Latest PNGs ==="
PNG_DIR=$(find "$OUTPUT_DIR" -type d -name "pngs" -maxdepth 2 2>/dev/null | head -1)
if [ -n "$PNG_DIR" ]; then
    find "$PNG_DIR" -name "*.png" -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF "  (" $5 ")"}'
else
    echo "  (no PNGs yet)"
fi

echo ""
echo "=== Log Tail ==="
LOG_FILE=$(find "$OUTPUT_DIR" -name "launch.log" -maxdepth 3 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    tail -8 "$LOG_FILE"
else
    echo "  (no log yet)"
fi

echo ""
echo "[fetch] Done at $(date '+%H:%M:%S')"
