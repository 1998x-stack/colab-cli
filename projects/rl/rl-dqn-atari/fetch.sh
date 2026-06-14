#!/bin/bash
# Fetch DQN Atari training results from Colab VM. Called by cron every 2 minutes.
# Usage: bash fetch.sh [session_name] [account]
set -euo pipefail

SESSION="${1:-dqn-atari}"
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
mkdir -p "$LOCAL_OUT"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FETCH $TIMESTAMP ==="

# Check session alive
if ! $COL sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[FATAL] Session '$SESSION' is DEAD or not found."
    $COL sessions 2>/dev/null || echo "  (no active sessions)"
    exit 1
fi
echo "  Session '$SESSION' alive."

# Download artifacts
$COL download -s "$SESSION" /content/dqn-output/metrics.json "$LOCAL_OUT/metrics.json" 2>&1 || echo "  metrics download skipped"
$COL download -s "$SESSION" /content/dqn-output/summary.json "$LOCAL_OUT/summary.json" 2>&1 || echo "  summary download skipped"

# Log
echo ""
echo "--- Log tail ---"
$COL download -s "$SESSION" /content/dqn_train.log "$LOCAL_OUT/train.log" 2>&1 || echo "  log download skipped"
if [ -f "$LOCAL_OUT/train.log" ]; then
    tail -10 "$LOCAL_OUT/train.log"
fi

# Metrics
echo ""
if [ -f "$LOCAL_OUT/summary.json" ]; then
    echo "--- Summary ---"
    python3 -c "
import json
with open('$LOCAL_OUT/summary.json') as f:
    s = json.load(f)
print(f\"  Env: {s.get('env','?')}  Device: {s.get('device','?')}\")
print(f\"  Episodes: {s.get('total_episodes',0)}  Steps: {s.get('total_steps',0)}\")
print(f\"  Best return: {s.get('best_return','?')}  Solved: {s.get('solved',False)}\")
print(f\"  Train time: {s.get('train_time_seconds',0)/60:.1f} min\")
" 2>/dev/null || echo "  (summary parse failed)"
fi

echo ""
echo "=== DONE $TIMESTAMP ==="
