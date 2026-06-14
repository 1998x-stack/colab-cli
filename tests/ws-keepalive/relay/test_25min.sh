#!/bin/bash
# One-shot 25-minute Colab GPU relay test.
#
# Usage: bash test_25min.sh <session_name> [account_suffix]
#   bash test_25min.sh relay-25min     # default account
#   bash test_25min.sh relay-25min cb  # account-b (stefaniehu929)
#
# Each watchdog is launched with nohup + & so they survive shell exit.
# Uses Config B proxy (HTTP CONNECT tunnel) for reliability from China.

set -e

SESSION="${1:?Usage: $0 <session_name> [account_suffix]}"
ACCOUNT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Account prefix
if [ -n "$ACCOUNT" ]; then
    ACCT_HOME="$HOME/colab-accounts/account-$ACCOUNT"
    COLAB="env HOME=$ACCT_HOME colab"
else
    COLAB="colab"
fi

# ── Proxy: Config B (HTTP CONNECT tunnel) ─────────────────
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
export no_proxy=""

echo "=== Colab 25-Minute GPU Relay Test ==="
echo "Session: $SESSION  Account: ${ACCOUNT:-default}"

# ── Step 1: Provision GPU session ─────────────────────────
echo ""
echo "[Step 1] Creating GPU session..."
$COLAB new --gpu T4 -s "$SESSION"
echo "Session created."

# ── Step 2: Upload training script ────────────────────────
echo ""
echo "[Step 2] Uploading training script..."
$COLAB upload "$SCRIPT_DIR/fake_train_25min.py" /content/fake_train_25min.py
echo "Uploaded."

# ── Step 3: Launch ws-1 (spawns training + 7-min watchdog) ─
echo ""
echo "[Step 3] Launching ws-1 (T+0 min) — spawns training + 7-min watchdog..."
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/launch_train.py" --timeout 540 \
    > /dev/null 2>&1 &
echo "ws-1 PID=$!"

# ── Step 4: Launch ws-2 at T+6 min ────────────────────────
echo ""
echo "[Step 4] Launching ws-2 at T+6 min..."
sleep 360
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/watchdog.py" --timeout 540 \
    > /dev/null 2>&1 &
echo "ws-2 PID=$!"

# ── Step 5: Launch ws-3 at T+12 min ───────────────────────
echo ""
echo "[Step 5] Launching ws-3 at T+12 min..."
sleep 360
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/watchdog.py" --timeout 540 \
    > /dev/null 2>&1 &
echo "ws-3 PID=$!"

# ── Step 6: Launch ws-4 at T+18 min ───────────────────────
echo ""
echo "[Step 6] Launching ws-4 at T+18 min..."
sleep 360
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/watchdog.py" --timeout 540 \
    > /dev/null 2>&1 &
echo "ws-4 PID=$!"

# ── Step 7: Launch ws-5 at T+24 min (buffer) ──────────────
echo ""
echo "[Step 7] Launching ws-5 at T+24 min (buffer)..."
sleep 360
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/watchdog.py" --timeout 540 \
    > /dev/null 2>&1 &
echo "ws-5 PID=$!"

# ── Step 8: Wait for training to complete ─────────────────
echo ""
echo "[Step 8] Waiting 2 min for training to finish (25 min total)..."
sleep 120

# ── Step 9: Download results ──────────────────────────────
mkdir -p "$OUTPUT_DIR"
echo ""
echo "[Step 9] Downloading results..."
$COLAB download -s "$SESSION" /content/relay-test-output/logs/train.log "$OUTPUT_DIR/train.log" 2>/dev/null || true
$COLAB download -s "$SESSION" /content/relay-test-output/logs/watchdog.log "$OUTPUT_DIR/watchdog.log" 2>/dev/null || true

# ── Step 10: Report ───────────────────────────────────────
echo ""
echo "=== Results ==="
if [ -f "$OUTPUT_DIR/train.log" ]; then
    LINES=$(wc -l < "$OUTPUT_DIR/train.log")
    echo "Train log: $LINES lines"
    echo "  First: $(head -1 "$OUTPUT_DIR/train.log" | cut -c1-120)"
    echo "  Last:  $(tail -1 "$OUTPUT_DIR/train.log" | cut -c1-120)"
    if grep -q "TRAIN_COMPLETE" "$OUTPUT_DIR/train.log"; then
        echo ""
        ELAPSED=$(grep "total_elapsed" "$OUTPUT_DIR/train.log" | tail -1)
        echo "  *** SUCCESS! $ELAPSED"
    else
        echo ""
        echo "  *** FAILED: Training did not complete"
    fi
else
    echo "  *** No train log downloaded"
fi

if [ -f "$OUTPUT_DIR/watchdog.log" ]; then
    LINES=$(wc -l < "$OUTPUT_DIR/watchdog.log")
    COMPLETIONS=$(grep -c "EXIT" "$OUTPUT_DIR/watchdog.log" || true)
    echo ""
    echo "Watchdog log: $LINES lines, $COMPLETIONS completions"
    grep "EXIT" "$OUTPUT_DIR/watchdog.log" | while read -r line; do
        echo "  $(echo "$line" | cut -c1-120)"
    done
fi

echo ""
echo "Output: $OUTPUT_DIR/"
echo "Session still alive: $SESSION (stop with: $COLAB stop -s $SESSION)"
