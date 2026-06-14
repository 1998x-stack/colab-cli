#!/bin/bash
# Fetch DDPG training outputs from Colab VM to local project directory.
# Called by cron every 2 minutes.
#
# Usage: bash fetch.sh [session_name] [account]
#   session_name: Colab session name (default: ddpg)
#   account: colab|cb|cc|clb (default: colab)

set -euo pipefail

SESSION="${1:-ddpg}"
ACCOUNT="${2:-colab}"

case "$ACCOUNT" in
    colab) COL="colab" ;;
    cb)    COL="cb" ;;
    cc)    COL="cc" ;;
    clb)   COL="clb" ;;
    *)     echo "ERROR: unknown account: $ACCOUNT"; exit 2 ;;
esac

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/rl/ddpg-gym/output"
mkdir -p "$OUT_DIR/plots"

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
$COL download -s "$SESSION" /content/ddpg-output/train.log "$OUT_DIR/train.log" 2>&1 || echo "  log download skipped (may not exist yet)"

# Download metrics
$COL download -s "$SESSION" /content/ddpg-output/metrics.json "$OUT_DIR/metrics.json" 2>&1 || echo "  metrics download skipped"

# Download progress plot
$COL download -s "$SESSION" /content/ddpg-output/plots/progress.png "$OUT_DIR/plots/progress_${TIMESTAMP}.png" 2>&1 || echo "  plot download skipped"

# Also copy latest as fixed name for easy viewing
if [ -f "$OUT_DIR/plots/progress_${TIMESTAMP}.png" ]; then
    cp "$OUT_DIR/plots/progress_${TIMESTAMP}.png" "$OUT_DIR/plots/progress_latest.png"
fi

# Download summary
$COL download -s "$SESSION" /content/ddpg-output/summary.json "$OUT_DIR/summary.json" 2>&1 || echo "  summary download skipped"

# Show summary
echo ""
echo "--- Summary ---"
if [ -f "$OUT_DIR/summary.json" ]; then
    python3 -c "
import json
with open('$OUT_DIR/summary.json') as f:
    s = json.load(f)
print(f\"  Env: {s.get('env','?')}  Device: {s.get('device','?')}\")
print(f\"  Episodes: {s.get('episodes_completed',0)}  Steps: {s.get('total_steps',0)}\")
print(f\"  Best eval: {s.get('best_eval_reward','?'):.2f}  Evals: {s.get('n_evals',0)}\")
" 2>/dev/null || echo "  (summary parse failed)"
fi

echo ""
if [ -f "$OUT_DIR/train.log" ]; then
    echo "--- Log tail ---"
    tail -5 "$OUT_DIR/train.log"
fi

if [ -f "$OUT_DIR/metrics.json" ]; then
    echo ""
    echo "--- Metrics ---"
    python3 -c "
import json
with open('$OUT_DIR/metrics.json') as f:
    m = json.load(f)
eps = m.get('episodes', [])
evals = m.get('eval_episodes', [])
print(f'Episodes: {len(eps)} | Evals: {len(evals)}')
if eps:
    last = eps[-1]
    print(f'Last: ep {last[\"episode\"]} reward={last[\"reward\"]:.2f}')
if evals:
    last_ev = evals[-1]
    print(f'Last eval: ep {last_ev[\"episode\"]} mean={last_ev[\"mean_reward\"]:.2f} ± {last_ev[\"std_reward\"]:.2f}')
" 2>&1 || true
fi

echo "=== DONE $TIMESTAMP ==="
