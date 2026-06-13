#!/bin/bash
# fetch.sh — Download training artifacts from Colab VM, extract, report.
# Called by cron watchtower every 2-3 minutes.
set -euo pipefail

SESSION="seq2seq"
PROJDIR="projects/seq2seq-t4"
OUTDIR="$PROJDIR/output"
mkdir -p "$OUTDIR"

# Proxy (explicit per-variable — never expand via $VAR)
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# --- 1. Check session alive ---
if ! colab sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[fetch] SESSION '$SESSION' NOT FOUND — may have expired"
    exit 1
fi

# --- 2. Tar outputs on VM ---
echo 'import subprocess, os; d="/content/seq2seq-t4"; subprocess.run(["tar","-czf","/content/seq2seq-t4-output.tar.gz","-C","/content","seq2seq-t4"], check=True) if os.path.exists(d) else print("NO_OUTDIR")' | colab exec -s "$SESSION" --timeout 15 2>/dev/null || true

# --- 3. Download tar ---
colab download -s "$SESSION" /content/seq2seq-t4-output.tar.gz "$OUTDIR/output.tar.gz" 2>/dev/null || {
    echo "[fetch] Download failed — WebSocket may be down, trying direct REST..."
    # Fallback: download individual known files
    colab download -s "$SESSION" /content/seq2seq-t4/logs/train.log "$OUTDIR/train.log" 2>/dev/null || true
    colab download -s "$SESSION" /content/seq2seq-t4/metrics.csv "$OUTDIR/metrics.csv" 2>/dev/null || true
}

# --- 4. Extract ---
if [ -f "$OUTDIR/output.tar.gz" ]; then
    tar -xzf "$OUTDIR/output.tar.gz" -C "$OUTDIR" 2>/dev/null || true
fi

# --- 5. Report ---
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  $(date '+%Y-%m-%d %H:%M:%S')  |  Session: $SESSION"
echo "══════════════════════════════════════════════════════════"

# Log tail
LOG="$OUTDIR/seq2seq-t4/logs/train.log"
if [ -f "$LOG" ]; then
    echo ""
    echo "── Log tail ──"
    tail -12 "$LOG"
else
    echo "(no log yet)"
fi

# Metrics
CSV="$OUTDIR/seq2seq-t4/metrics.csv"
if [ -f "$CSV" ]; then
    echo ""
    echo "── Latest metrics ──"
    tail -3 "$CSV"
fi

# Checkpoints
CKPT="$OUTDIR/seq2seq-t4/checkpoints"
if [ -d "$CKPT" ]; then
    echo ""
    echo "── Checkpoints ──"
    ls -lh "$CKPT"/
fi

# Training curves (file existence)
PNGS="$OUTDIR/seq2seq-t4/pngs"
if [ -d "$PNGS" ]; then
    echo ""
    echo "── Figures ──"
    ls -lh "$PNGS"/
fi

echo ""
echo "── Local output: $OUTDIR ──"
echo ""
