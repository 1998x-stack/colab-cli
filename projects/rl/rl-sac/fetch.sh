#!/bin/bash
# Fetch SAC training results from Colab VM. Called by cron every 2 minutes.
# Usage: bash fetch.sh [session_name] [account]
set -euo pipefail

SESSION="${1:-sac}"
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

# Download log
$COL download -s "$SESSION" /content/sac_train.log "$LOCAL_OUT/train.log" 2>&1 || echo "  log download skipped"

# Download summary
$COL download -s "$SESSION" /content/sac-summary.json "$LOCAL_OUT/summary.json" 2>&1 || echo "  summary download skipped"

# Show results
echo ""
if [ -f "$LOCAL_OUT/train.log" ]; then
    echo "--- Log tail ---"
    tail -10 "$LOCAL_OUT/train.log"
fi

echo ""
if [ -f "$LOCAL_OUT/summary.json" ]; then
    echo "--- Summary ---"
    python3 -c "
import json
with open('$LOCAL_OUT/summary.json') as f:
    s = json.load(f)
print(f\"  Env: {s.get('env','?')}  Device: {s.get('device','?')}\")
print(f\"  Episodes: {s.get('episodes_completed',0)}  Steps: {s.get('total_steps',0)}\")
print(f\"  Best return: {s.get('best_return',0):.2f}  Final avg100: {s.get('final_avg100',0):.2f}\")
" 2>/dev/null || echo "  (summary parse failed)"
fi

echo ""
echo "=== DONE $TIMESTAMP ==="
