#!/bin/bash
# Fetch DDPG training outputs from Colab VM to local project directory.
# Called by cron every 2 minutes.

set -euo pipefail

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/ddpg-gym/output"
mkdir -p "$OUT_DIR/plots"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FETCH $TIMESTAMP ==="

# Download log
colab download -s ddpg /content/ddpg-output/train.log "$OUT_DIR/train.log" 2>&1 || echo "  log download skipped (may not exist yet)"

# Download metrics
colab download -s ddpg /content/ddpg-output/metrics.json "$OUT_DIR/metrics.json" 2>&1 || echo "  metrics download skipped"

# Download progress plot
colab download -s ddpg /content/ddpg-output/plots/progress.png "$OUT_DIR/plots/progress_${TIMESTAMP}.png" 2>&1 || echo "  plot download skipped"

# Also copy latest as fixed name for easy viewing
if [ -f "$OUT_DIR/plots/progress_${TIMESTAMP}.png" ]; then
    cp "$OUT_DIR/plots/progress_${TIMESTAMP}.png" "$OUT_DIR/plots/progress_latest.png"
fi

# Show summary
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
