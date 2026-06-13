#!/bin/bash
# Fetch DDQN vs NoisyNet training outputs from Colab VM.
set -euo pipefail

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/ddqn-noisy-ram/output"
mkdir -p "$OUT_DIR/plots"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

SESSION="${COLAB_SESSION:-ddqn}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FETCH $TIMESTAMP ==="

colab download -s "$SESSION" /content/ddqn-noisy-output/train.log "$OUT_DIR/train.log" 2>&1 || echo "  log download skipped"
colab download -s "$SESSION" /content/ddqn-noisy-output/metrics.json "$OUT_DIR/metrics.json" 2>&1 || echo "  metrics download skipped"
colab download -s "$SESSION" /content/ddqn-noisy-output/plots/progress.png "$OUT_DIR/plots/progress_${TIMESTAMP}.png" 2>&1 || echo "  plot download skipped"
colab download -s "$SESSION" /content/ddqn-noisy-output/plots/comparison.png "$OUT_DIR/plots/comparison_${TIMESTAMP}.png" 2>&1 || echo "  comparison download skipped"

if [ -f "$OUT_DIR/plots/progress_${TIMESTAMP}.png" ]; then
    cp "$OUT_DIR/plots/progress_${TIMESTAMP}.png" "$OUT_DIR/plots/progress_latest.png"
fi
if [ -f "$OUT_DIR/plots/comparison_${TIMESTAMP}.png" ]; then
    cp "$OUT_DIR/plots/comparison_${TIMESTAMP}.png" "$OUT_DIR/plots/comparison_latest.png"
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
for name, data in sorted(m.items()):
    eps = data.get('episodes', [])
    evals = data.get('evals', [])
    last_r = eps[-1]['reward'] if eps else float('nan')
    best_ev = max(e['mean_reward'] for e in evals) if evals else float('nan')
    print(f'  {name:30s}  eps={len(eps):4d}  last_r={last_r:8.1f}  best_ev={best_ev:8.1f}')
" 2>&1 || true
fi

echo "=== DONE $TIMESTAMP ==="
