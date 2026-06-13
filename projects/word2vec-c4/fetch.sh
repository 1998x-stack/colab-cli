#!/bin/bash
# Fetch Word2Vec training results from Colab VM (REST download, survives WebSocket drops).
# Called by cron every 2 minutes. Excludes checkpoints from download.
#
# Usage: ./fetch.sh [session_name]
#
# Uses two proxy configs:
#   Step 1 (exec tar): Config B — HTTP CONNECT for WebSocket stability
#   Step 2 (download): Config A — SOCKS5 + no_proxy to bypass proxy for *.colab.dev

set -euo pipefail

SESSION="${1:-word2vec-c4}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
ACCT_HOME=~/colab-accounts/account-b
COLB_BIN=/Users/mx/.local/bin/colab

mkdir -p "$LOCAL_OUT"

# Step 1: Tar output on VM (excluding checkpoints) — use Config B for exec stability
echo "[fetch] $(date '+%H:%M:%S') Tarring output on VM..."
(
    export HTTPS_PROXY=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    export ALL_PROXY=socks5://127.0.0.1:7890

    echo '
import subprocess as s
r = s.run([
    "tar", "-czf", "/content/word2vec-c4-output.tar.gz",
    "-C", "/content/word2vec-c4-output",
    "--exclude=checkpoints",
    "."
], capture_output=True)
if r.returncode != 0:
    print(f"tar stderr: {r.stderr.decode()}")
' | HOME="$ACCT_HOME" "$COLB_BIN" exec -s "$SESSION" --timeout 30 2>/dev/null
) || {
    echo "[fetch] WARNING: exec tar failed (WebSocket may be down)"
}

# Step 2: Download tar — use Config A (SOCKS5 + no_proxy for direct download)
echo "[fetch] Downloading..."
(
    export HTTPS_PROXY=socks5://127.0.0.1:7890
    export HTTP_PROXY=socks5://127.0.0.1:7890
    export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

    HOME="$ACCT_HOME" "$COLB_BIN" download -s "$SESSION" \
        /content/word2vec-c4-output.tar.gz \
        "$LOCAL_OUT/output.tar.gz" 2>/dev/null
) || {
    echo "[fetch] WARNING: download failed — session may be dead or tar missing"
    exit 0
}

# Step 3: Extract (tar created with -C /content/word2vec-c4-output, so extracts flat)
cd "$LOCAL_OUT"
tar -xzf output.tar.gz 2>/dev/null || {
    echo "[fetch] WARNING: extract failed"
    exit 0
}

# Step 4: Report
echo "[fetch] $(date '+%H:%M:%S') Done."

if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""
    echo "── Last 8 log lines ──"
    tail -8 "$LOCAL_OUT/logs/train.log"
fi

if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""
    echo "── Metrics CSV tail (last 3 rows) ──"
    head -1 "$LOCAL_OUT/metrics.csv"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi

echo ""
echo "── PNGs ──"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"

echo ""
echo "Files in: $LOCAL_OUT"
