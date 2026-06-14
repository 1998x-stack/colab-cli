#!/bin/bash
# Fetch PPO MuJoCo training results from Colab VM. Called by cron every 2 minutes.
# Usage: bash fetch.sh [session_name] [account]
set -euo pipefail
SESSION="${1:-ppo-mujoco}"
ACCOUNT="${2:-colab}"

case "$ACCOUNT" in
    colab) COL="colab" ;;
    cb)    COL="cb" ;;
    cc)    COL="cc" ;;
    clb)   COL="clb" ;;
    *)     echo "ERROR: unknown account: $ACCOUNT"; exit 2 ;;
esac

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="ppo-mujoco-output.tar.gz"
mkdir -p "$LOCAL_OUT"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# Check session alive
if ! $COL sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[FATAL] Session '$SESSION' is DEAD or not found."
    $COL sessions 2>/dev/null || echo "  (no active sessions)"
    exit 1
fi
echo "[fetch] $(date '+%H:%M:%S') Session alive. Downloading..."
"$COL" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead"
    exit 0
}

cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" --strip-components=1 2>/dev/null || { echo "[fetch] WARNING: extract failed"; exit 0; }
echo "[fetch] $(date '+%H:%M:%S') Done."

if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo "══ Last 8 log lines ══"
    tail -8 "$LOCAL_OUT/logs/train.log"
fi
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""; echo "══ Metrics CSV tail ══"
    head -1 "$LOCAL_OUT/metrics.csv"; tail -3 "$LOCAL_OUT/metrics.csv"
fi
echo ""; echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"
echo ""; echo "Files in: $LOCAL_OUT"
