#!/bin/bash
# fetch.sh — cron task: download Colab outputs via REST, print status.
#
# Usage:
#   SESSION_NAME=my-training PROJECT=my-project bash fetch.sh
#   SESSION_NAME=my-training PROJECT=my-project ACCOUNT=cb bash fetch.sh
#
# Env vars:
#   SESSION_NAME  — colab session name (required)
#   PROJECT       — project name for output dir matching (default: SESSION_NAME)
#   ACCOUNT       — which colab account: colab|cb|cc|clb (default: colab)
#   OUT_ROOT      — local output root (default: ./output)
#   COLAB_BIN     — path to colab binary (default: ~/.local/bin/colab)
set -euo pipefail

SESSION="${SESSION_NAME:?must set SESSION_NAME}"
PROJECT="${PROJECT:-$SESSION}"
ACCOUNT="${ACCOUNT:-colab}"
OUT_ROOT="${OUT_ROOT:-./output}"
COLAB_BIN="${COLAB_BIN:-$HOME/.local/bin/colab}"

# ── Account setup ──────────────────────────────────────
case "$ACCOUNT" in
    cb)  ACCT_HOME="$HOME/colab-accounts/account-b" ;;
    cc)  ACCT_HOME="$HOME/colab-accounts/account-c" ;;
    clb) ACCT_HOME="$HOME/colab-accounts/account-clb" ;;
    *)   ACCT_HOME="$HOME" ;;
esac

COLAB="HOME=$ACCT_HOME $COLAB_BIN"

# ── Proxy (Config B — HTTP CONNECT, most reliable) ─────
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

REMOTE_DIR="/content/${PROJECT}-output"
LOCAL_DIR="$OUT_ROOT/${PROJECT}-output"
TAR_NAME="fetch-$(date -u '+%H%M%S').tar.gz"

mkdir -p "$OUT_ROOT"

echo "=== [$(date '+%H:%M:%S')] Fetch: $SESSION (project=$PROJECT, account=$ACCOUNT) ==="

# ═══════════════════════════════════════════════════════════════
# Step 1: Session health check
# ═══════════════════════════════════════════════════════════════
echo "--- Session check ---"
SESSION_OUTPUT=$($COLAB sessions 2>&1) || true

if echo "$SESSION_OUTPUT" | grep -q "No active sessions"; then
    echo ""
    echo "!!! SESSION DEAD — no active sessions on server !!!"
    echo "DEAD at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$OUT_ROOT/death_notice.txt"

    # Print whatever we have locally
    for f in watchdog.log train.log; do
        if [ -f "$LOCAL_DIR/logs/$f" ] && [ -s "$LOCAL_DIR/logs/$f" ]; then
            echo "=== Last known $f ==="
            tail -5 "$LOCAL_DIR/logs/$f"
        fi
    done
    if [ -f "$LOCAL_DIR/metrics.csv" ] && [ -s "$LOCAL_DIR/metrics.csv" ]; then
        echo "=== Last metrics row ==="
        tail -1 "$LOCAL_DIR/metrics.csv"
    fi
    exit 0
fi

# Extract session runtime if present
RUNTIME=$(echo "$SESSION_OUTPUT" | grep "$SESSION" | grep -oE '[0-9]+:[0-9]+:[0-9]+' | head -1 || true)
if [ -n "$RUNTIME" ]; then
    echo "  Session alive: $SESSION (runtime: $RUNTIME)"
else
    echo "  Session alive: $SESSION"
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Tar outputs on VM (exclude checkpoints — they're big)
# ═══════════════════════════════════════════════════════════════
echo "--- Tar on VM ---"
TAR_SCRIPT=$(cat <<PYEOF
import subprocess, os
remote = "$REMOTE_DIR"
if not os.path.isdir(remote):
    print(f"SKIP: {remote} does not exist on VM")
else:
    subprocess.run(["tar", "-czf", "/content/output.tar.gz",
                    "--exclude=checkpoints", "-C", "/content",
                    f"${PROJECT}-output"],
                   check=True, timeout=30)
    size = os.path.getsize("/content/output.tar.gz")
    print(f"OK: /content/output.tar.gz ({size:,} bytes)")
PYEOF
)

TAR_OUTPUT=$(echo "$TAR_SCRIPT" | $COLAB exec -s "$SESSION" --timeout 45 2>&1) || true
echo "$TAR_OUTPUT"

if echo "$TAR_OUTPUT" | grep -q "SKIP"; then
    echo "  (output dir not yet created on VM — training may not have started)"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# Step 3: Download tar via REST
# ═══════════════════════════════════════════════════════════════
echo "--- Downloading ---"
REMOTE_TAR="/content/output.tar.gz"
LOCAL_TAR="$OUT_ROOT/$TAR_NAME"

if $COLAB download -s "$SESSION" "$REMOTE_TAR" "$LOCAL_TAR" 2>&1; then
    TAR_SIZE=$(ls -lh "$LOCAL_TAR" | awk '{print $5}')
    echo "  Downloaded: $TAR_NAME ($TAR_SIZE)"
else
    echo "  Download failed — falling back to individual files"
    for f in logs/train.log logs/watchdog.log metrics.csv; do
        $COLAB download -s "$SESSION" "$REMOTE_DIR/$f" "$LOCAL_DIR/$f" 2>&1 || \
            echo "  (download $f failed — file may not exist)"
    done
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: Extract
# ═══════════════════════════════════════════════════════════════
if [ -f "$LOCAL_TAR" ] && [ -s "$LOCAL_TAR" ]; then
    echo "--- Extracting ---"
    mkdir -p "$LOCAL_DIR"
    tar -xzf "$LOCAL_TAR" -C "$OUT_ROOT" 2>&1 || echo "  (extract had warnings — partial data OK)"
    rm -f "$LOCAL_TAR"
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: Report
# ═══════════════════════════════════════════════════════════════

# --- Watchdog log ---
echo ""
echo "=== WATCHDOG LOG (last 5 lines) ==="
WD_LOG="$LOCAL_DIR/logs/watchdog.log"
if [ -f "$WD_LOG" ] && [ -s "$WD_LOG" ]; then
    tail -5 "$WD_LOG"
    echo "  ($(wc -l < "$WD_LOG" | tr -d ' ') lines total)"
else
    echo "  (no watchdog log yet)"
fi

# --- Training log ---
echo ""
echo "=== TRAIN LOG (last 8 lines) ==="
TRAIN_LOG="$LOCAL_DIR/logs/train.log"
if [ -f "$TRAIN_LOG" ] && [ -s "$TRAIN_LOG" ]; then
    tail -8 "$TRAIN_LOG"
    echo "  ($(wc -l < "$TRAIN_LOG" | tr -d ' ') lines total)"
else
    echo "  (no train log yet)"
fi

# --- Metrics CSV ---
echo ""
echo "=== METRICS CSV (last 3 rows) ==="
METRICS="$LOCAL_DIR/metrics.csv"
if [ -f "$METRICS" ] && [ -s "$METRICS" ]; then
    head -1 "$METRICS"
    echo "  ..."
    tail -3 "$METRICS"
    echo "  ($(wc -l < "$METRICS" | tr -d ' ') rows total)"
else
    echo "  (no metrics yet)"
fi

# --- PNGs ---
echo ""
echo "=== PNGs ==="
PNG_DIR="$LOCAL_DIR/pngs"
if [ -d "$PNG_DIR" ] && [ "$(ls -A "$PNG_DIR" 2>/dev/null)" ]; then
    for png in "$PNG_DIR"/*.png; do
        if [ -f "$png" ]; then
            ls -lh "$png" | awk '{print "  " $NF " (" $5 ", modified " $6 " " $7 " " $8 ")"}'
        fi
    done
else
    echo "  (no PNGs yet)"
fi

# --- Summary ---
echo ""
echo "=== Status at $(date -u '+%H:%M:%S UTC') ===="
echo "fetch complete"
