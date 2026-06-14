#!/bin/bash
# fetch.sh — cron task: check session health, download logs, print status
# Usage: SESSION_NAME=ws-test bash fetch.sh
set -euo pipefail

SESSION="${SESSION_NAME:-ws-test}"
OUT_DIR="/Users/mx/Desktop/projects/colab-cli/tests/ws-keepalive/output"
REMOTE_DIR="/content/ws-test-output"

mkdir -p "$OUT_DIR"

echo "=== [$(date '+%H:%M:%S')] Fetch: $SESSION ==="

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# Step 1: Check if session is alive
echo "--- Session check ---"
SESSION_OUTPUT=$(HOME=~/colab-accounts/account-c /Users/mx/.local/bin/colab sessions 2>&1 || true)
echo "$SESSION_OUTPUT"

if echo "$SESSION_OUTPUT" | grep -q "No active sessions"; then
    echo ""
    echo "!!! SESSION DEAD — no active sessions on server !!!"
    echo "DEAD at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$OUT_DIR/death_notice.txt"
    # Print whatever logs we have
    for logfile in watchdog.log train.log; do
        if [ -f "$OUT_DIR/$logfile" ]; then
            echo "=== Last known $logfile ==="
            tail -5 "$OUT_DIR/$logfile"
        fi
    done
    exit 0
fi

# Step 2: Download logs directly (REST, works even while exec is running)
echo "--- Downloading logs ---"
for logfile in watchdog.log train.log; do
    HOME=~/colab-accounts/account-c /Users/mx/.local/bin/colab download \
        -s "$SESSION" \
        "$REMOTE_DIR/logs/$logfile" \
        "$OUT_DIR/$logfile" \
        2>&1 || echo "  (download $logfile failed — file may not exist yet)"
done

# Step 3: Print log tails
echo ""
echo "=== WATCHDOG LOG (last 8 lines) ==="
if [ -f "$OUT_DIR/watchdog.log" ] && [ -s "$OUT_DIR/watchdog.log" ]; then
    tail -8 "$OUT_DIR/watchdog.log"
    echo "  ($(wc -l < "$OUT_DIR/watchdog.log") lines total)"
else
    echo "  (no watchdog log yet — launcher may still be starting)"
fi

echo ""
echo "=== TRAIN LOG (last 8 lines) ==="
if [ -f "$OUT_DIR/train.log" ] && [ -s "$OUT_DIR/train.log" ]; then
    tail -8 "$OUT_DIR/train.log"
    echo "  ($(wc -l < "$OUT_DIR/train.log") lines total)"
else
    echo "  (no train log yet)"
fi

echo ""
echo "=== Status at $(date -u '+%H:%M:%S UTC') ==="
echo "--- fetch complete ---"
