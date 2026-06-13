#!/bin/bash
# Fetch RAG FastText training results from Colab VM. Called by cron every 3 minutes.
set -euo pipefail
SESSION="${1:-rag-fasttext}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="rag-fasttext-output.tar.gz"
mkdir -p "$LOCAL_OUT"

export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890
COLB="$(which colab)"

echo "[fetch] $(date '+%H:%M:%S') Starting fetch for session: $SESSION"

# 1. Check session alive
echo "[fetch] Checking session ..."
"$COLB" sessions 2>/dev/null | grep -q "$SESSION" || {
    echo "[fetch] WARNING: session $SESSION not found — may be dead"
    exit 0
}

# 2. Tar output on VM (exclude checkpoints and training temp files)
echo "[fetch] Tarring output on VM ..."
echo '
import subprocess, os
out = "/content/rag-fasttext-output"
tar = "/content/rag-fasttext-output.tar.gz"
subprocess.run(["tar", "-czf", tar, "-C", out,
    "--exclude=checkpoints", "--exclude=fasttext_train.txt",
    "--exclude=*.bin", "--exclude=*.pt",
    "."], check=True)
print(f"Tarball: {os.path.getsize(tar)/1024:.0f} KB")
' | "$COLB" exec -s "$SESSION" --timeout 15 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed, trying direct download ..."
}

# 3. Download
echo "[fetch] Downloading ..."
"$COLB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or output not ready"
    exit 0
}

# 4. Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" 2>/dev/null || { echo "[fetch] WARNING: extract failed"; exit 0; }
echo "[fetch] $(date '+%H:%M:%S') Done."

# 5. Report
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""; echo "══ Last 10 log lines ══"
    tail -10 "$LOCAL_OUT/logs/train.log"
fi
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""; echo "══ Metrics ══"
    cat "$LOCAL_OUT/metrics.csv"
fi
if [ -f "$LOCAL_OUT/index_info.json" ]; then
    echo ""; echo "══ Index Info ══"
    python3 -c "import json; d=json.load(open('$LOCAL_OUT/index_info.json')); print(json.dumps(d.get('metrics',{}), indent=2))" 2>/dev/null || true
fi
echo ""; echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"
echo ""; echo "Files in: $LOCAL_OUT"
