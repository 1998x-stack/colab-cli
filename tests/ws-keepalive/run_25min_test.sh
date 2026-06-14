#!/bin/bash
# 25-min Colab GPU relay test — tight windows, short overlap.
# Run in background: bash run_25min_test.sh &
# Logs: /tmp/relay25-*.log
set -e

SESSION="relay25"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/relay25-orch.log"

log() { echo "[$(date -u '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Proxy Config B
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

log "=== 25-Minute Relay Test ==="
log "Params: 5-min windows, 30s overlap, 6 watchdogs"

# Cleanup any old session
colab stop -s "$SESSION" 2>/dev/null || true
sleep 2

# Create session
log "Creating GPU session..."
colab new --gpu T4 -s "$SESSION"
log "Session READY"
START_TIME=$(date -u +%s)

# Launch ws-1: spawns training + 5-min watchdog
log "T+0:00 — Launching ws-1 (spawns training + 5-min watchdog)"
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/ws1_5min.py" --timeout 420 \
    > /tmp/relay25-ws1.log 2>&1 &
WS1_PID=$!
log "ws-1 PID=$WS1_PID"

# Wait 30s to verify ws-1 started
sleep 30
if ! kill -0 $WS1_PID 2>/dev/null; then
    log "FATAL: ws-1 died immediately"
    tail -20 /tmp/relay25-ws1.log
    exit 1
fi
log "ws-1 verified running"

# Launch ws-2 at T+4:30 (30s before ws-1 exits at T+5:00)
log "T+4:30 — Launching ws-2"
sleep 240
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/wd_5min.py" --timeout 420 \
    > /tmp/relay25-ws2.log 2>&1 &
log "ws-2 PID=$!"

# Launch ws-3 at T+9:00 (30s before ws-2 exits at T+9:30)
log "T+9:00 — Launching ws-3"
sleep 270
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/wd_5min.py" --timeout 420 \
    > /tmp/relay25-ws3.log 2>&1 &
log "ws-3 PID=$!"

# Launch ws-4 at T+13:30
log "T+13:30 — Launching ws-4"
sleep 270
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/wd_5min.py" --timeout 420 \
    > /tmp/relay25-ws4.log 2>&1 &
log "ws-4 PID=$!"

# Launch ws-5 at T+18:00
log "T+18:00 — Launching ws-5"
sleep 270
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/wd_5min.py" --timeout 420 \
    > /tmp/relay25-ws5.log 2>&1 &
log "ws-5 PID=$!"

# Launch ws-6 at T+22:30 (buffer)
log "T+22:30 — Launching ws-6 (buffer)"
sleep 270
nohup colab exec -s "$SESSION" -f "$SCRIPT_DIR/wd_5min.py" --timeout 420 \
    > /tmp/relay25-ws6.log 2>&1 &
log "ws-6 PID=$!"

# Wait for training to complete (25 min total)
NOW=$(date -u +%s)
ELAPSED=$((NOW - START_TIME))
WAIT=$((1560 - ELAPSED))  # 26 min total - elapsed
if [ $WAIT -gt 0 ]; then
    log "Waiting ${WAIT}s for training to complete..."
    sleep $WAIT
fi
log "T+26:00 — Training should be done"

# Download results
OUTDIR="$SCRIPT_DIR/output/relay25"
mkdir -p "$OUTDIR"
log "Downloading results..."
colab download -s "$SESSION" /content/relay25-output/logs/train.log "$OUTDIR/train.log" 2>&1 | tee -a "$LOG" || true
colab download -s "$SESSION" /content/relay25-output/logs/relay.log "$OUTDIR/relay.log" 2>&1 | tee -a "$LOG" || true

# Verify
log "=== Results ==="
if [ -f "$OUTDIR/train.log" ]; then
    LINES=$(wc -l < "$OUTDIR/train.log")
    log "Train log: $LINES lines"
    log "  First: $(head -1 "$OUTDIR/train.log")"
    log "  Last:  $(tail -1 "$OUTDIR/train.log")"
    if grep -q "COMPLETE" "$OUTDIR/train.log"; then
        ELAPSED=$(grep "total" "$OUTDIR/train.log" | tail -1)
        log "*** SUCCESS! $ELAPSED"
    else
        log "*** PARTIAL: training did not complete"
    fi
else
    log "*** FAILED: no train log downloaded (session reclaimed)"
fi

if [ -f "$OUTDIR/relay.log" ]; then
    log ""
    log "Watchdog completions:"
    grep "EXIT" "$OUTDIR/relay.log" | while read -r line; do
        log "  $line"
    done
fi

log "Done. Output in $OUTDIR/"
log "Session: $SESSION (still alive? check: colab status -s $SESSION)"
