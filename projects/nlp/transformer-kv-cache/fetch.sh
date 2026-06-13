#!/bin/bash
# Fetch transformer KV cache training results from Colab VM. Called by cron every 2 minutes.
set -euo pipefail
SESSION="${1:-kv-cache}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="kv-cache-output.tar.gz"
mkdir -p "$LOCAL_OUT"

# Proxy (Config B)
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# Use clb account (xieminghack) — change ACCOUNT to switch
ACCOUNT="${2:-clb}"
if [ "$ACCOUNT" = "clb" ]; then
    export HOME="$HOME/colab-accounts/account-clb"
elif [ "$ACCOUNT" = "c" ]; then
    export HOME="$HOME/colab-accounts/account-c"
fi
COLB="$(which colab)"

echo "[fetch] $(date '+%H:%M:%S') Session: $SESSION"

# 1. Check session alive
"$COLB" sessions 2>/dev/null | grep -q "$SESSION" || {
    echo "[fetch] WARNING: session $SESSION not found — may be dead"
    exit 0
}

# 2. Tar output on VM (exclude heavy checkpoints)
echo '
import subprocess, os
out = "/content/transformer-kv-cache-output"
tar = "/content/kv-cache-output.tar.gz"
subprocess.run(["tar", "-czf", tar, "-C", out,
    "--exclude=checkpoints", "."], check=True)
print(f"Tarball: {os.path.getsize(tar)/1024:.0f} KB")
' | "$COLB" exec -s "$SESSION" --timeout 15 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed, trying direct download..."
}

# 3. Download
"$COLB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or output not ready"
    exit 0
}

# 4. Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" 2>/dev/null || { echo "[fetch] WARNING: extract failed"; exit 0; }
echo "[fetch] $(date '+%H:%M:%S') Extract done."

# 5. Report
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""; echo "══ Last 5 log lines ══"
    tail -5 "$LOCAL_OUT/logs/train.log"
fi
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""; echo "══ Last 3 metrics rows ══"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi
echo ""; echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"
echo ""; echo "[fetch] Done. Files in $LOCAL_OUT"
