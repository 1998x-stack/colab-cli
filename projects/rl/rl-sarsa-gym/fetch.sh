#!/bin/bash
# Fetch training results from Colab VM.
# Usage: ./fetch.sh [session_name]
# Called by cron every 2 minutes.

set -euo pipefail

SESSION="${1:-rl-sarsa}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"

mkdir -p "$LOCAL_OUT"

# Proxy setup (required from China)
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

COLB="$(which colab)"

# Step 1: Tar output on VM
echo "[fetch] $(date '+%H:%M:%S') Tarring output on VM..."
echo 'import subprocess as s; s.run(["tar","-czf","/content/rl-sarsa-output.tar.gz","-C","/content","rl-sarsa-output"], capture_output=True)' \
  | "$COLB" exec -s "$SESSION" --timeout 30 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed (WebSocket may be down), trying download directly..."
}

# Step 2: Download tar
echo "[fetch] Downloading..."
"$COLB" download -s "$SESSION" /content/rl-sarsa-output.tar.gz "$LOCAL_OUT/output.tar.gz" 2>/dev/null || {
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

# Print last 5 log lines
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo "── Last 5 log lines ──"
    tail -5 "$LOCAL_OUT/logs/train.log"
fi

# Print last 3 CSV rows
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""
    echo "── Metrics CSV tail ──"
    echo "episode,reward,steps,epsilon,avg100_reward,q_mean,q_max,elapsed_s,td_error_mean"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi

echo ""
echo "── PNGs ──"
ls -la "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"

echo ""
echo "Files in: $LOCAL_OUT"
