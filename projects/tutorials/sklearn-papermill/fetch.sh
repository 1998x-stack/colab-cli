#!/bin/bash
# fetch.sh — Download outputs from Colab session, print latest metrics.
#
# Usage: bash fetch.sh [session-name]
#   Default session: sklearn-tutorial
#
# Designed for cron watchtower: called every 2-5 min to pull incremental results.
# Handles: session health check, tar on VM, download, extract, report.

set -euo pipefail

SESSION="${1:-sklearn-tutorial}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
FETCH_DIR="$PROJ_DIR/fetched"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

mkdir -p "$FETCH_DIR"

echo "[fetch] $(date '+%H:%M:%S') — session=$SESSION"

# ---- 1. Health check ----
if ! colab sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[fetch] SESSION DEAD — $SESSION not found in colab sessions"
    echo "[fetch] Run: bash deploy.sh to re-provision"
    exit 1
fi

# ---- 2. Tar output on VM ----
echo "[fetch] Tarring output on VM..."
echo 'import subprocess, os; out="/content/tutorial-output"; subprocess.run(["tar","-czf","/content/tutorial-output.tar.gz","-C","/content","tutorial-output"]) if os.path.exists(out) else print("OUTPUT DIR MISSING — notebook may still be running")' | colab exec -s "$SESSION" --timeout 15 2>/dev/null || {
    echo "[fetch] tar via exec failed — trying direct download of known paths"
}

# ---- 3. Download tar ----
echo "[fetch] Downloading..."
colab download -s "$SESSION" /content/tutorial-output.tar.gz "$FETCH_DIR/output.tar.gz" 2>/dev/null || {
    echo "[fetch] Download failed — output dir may not exist yet (notebook still running?)"
    # Try downloading individual files as fallback
    for f in /content/papermill.log /content/tutorial-output/metrics.csv; do
        colab download -s "$SESSION" "$f" "$FETCH_DIR/$(basename $f)" 2>/dev/null || true
    done
}

# ---- 4. Extract ----
if [ -f "$FETCH_DIR/output.tar.gz" ]; then
    tar -xzf "$FETCH_DIR/output.tar.gz" -C "$FETCH_DIR/" 2>/dev/null || true
    echo "[fetch] Extracted to $FETCH_DIR/tutorial-output/"
fi

# ---- 5. Report ----
echo ""
echo "=== Latest metrics ==="
if [ -f "$FETCH_DIR/metrics.csv" ]; then
    echo "  CSV rows: $(wc -l < "$FETCH_DIR/metrics.csv" | tr -d ' ')"
    echo "  Last row:"
    tail -1 "$FETCH_DIR/metrics.csv"
else
    echo "  (no metrics.csv yet)"
fi

echo ""
echo "=== Latest log ==="
if [ -f "$FETCH_DIR/papermill.log" ]; then
    echo "  Log lines: $(wc -l < "$FETCH_DIR/papermill.log" | tr -d ' ')"
    echo "  --- tail ---"
    tail -8 "$FETCH_DIR/papermill.log"
elif [ -d "$FETCH_DIR/tutorial-output/logs" ]; then
    tail -10 "$FETCH_DIR/tutorial-output/logs/train.log"
else
    echo "  (no log available yet)"
fi

echo ""
echo "=== PNGs ==="
if [ -d "$FETCH_DIR/tutorial-output/pngs" ]; then
    ls -la "$FETCH_DIR/tutorial-output/pngs/"
else
    echo "  (no pngs yet)"
fi

echo ""
echo "[fetch] Done — $(date '+%H:%M:%S')"
