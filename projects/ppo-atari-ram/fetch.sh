#!/bin/bash
# Fetch PPO Atari RAM training results from Colab VM.
# Usage: ./fetch.sh [session_name]
# Called by cron every 2 minutes.
set -euo pipefail

SESSION="${1:-ppo-atari}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="ppo-atari-output.tar.gz"

mkdir -p "$LOCAL_OUT"

# Proxy setup
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

COLB="$(which colab)"

# Step 1: Tar output on VM
echo "[fetch] $(date '+%H:%M:%S') Tarring output on VM..."
echo 'import subprocess as s; s.run(["tar","-czf","/content/ppo-atari-output.tar.gz","-C","/content","ppo-atari-output"], capture_output=True)' \
  | "$COLB" exec -s "$SESSION" --timeout 30 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed (WebSocket may be down), trying download directly..."
}

# Step 2: Download tar
echo "[fetch] Downloading..."
"$COLB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or tar missing"
    exit 0
}

# Step 3: Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" --strip-components=1 2>/dev/null || {
    echo "[fetch] WARNING: extract failed"
    exit 0
}

# Step 4: Report
echo "[fetch] $(date '+%H:%M:%S') Done."

# Print last 8 log lines
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo "══ Last 8 log lines ══"
    tail -8 "$LOCAL_OUT/logs/train.log"
fi

# Print last 3 CSV rows
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""
    echo "══ Metrics CSV tail ══"
    head -1 "$LOCAL_OUT/metrics.csv"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi

# PNGs
echo ""
echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"

echo ""
echo "Files in: $LOCAL_OUT"
