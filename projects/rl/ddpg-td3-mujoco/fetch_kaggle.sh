#!/bin/bash
# Check DDPG vs TD3 MuJoCo training progress on Kaggle.
# Called by cron every 2 minutes.
set -euo pipefail

KERNEL="xieming1998/ddpg-td3-mujoco"
OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/ddpg-td3-mujoco/output"
TOKEN="$(cat /Users/mx/Desktop/projects/colab-cli/.kaggle/access_token4)"
export KAGGLE_API_TOKEN="$TOKEN"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== KAGGLE CHECK $TIMESTAMP ==="

# 1. Status
STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
echo "Status: $STATUS"

# 2. Logs (may be empty due to buffering on GPU+internet kernels)
echo ""
echo "--- Log tail ---"
kaggle kernels logs "$KERNEL" 2>&1 | tail -20 || echo "  logs unavailable"

# 3. If complete or error, download outputs
if echo "$STATUS" | grep -q "complete\|error\|cancel"; then
    echo ""
    echo "=== Kernel finished! Downloading outputs ==="
    mkdir -p "$OUT_DIR"
    kaggle kernels output "$KERNEL" -p "$OUT_DIR" 2>&1 || echo "  output download failed"

    # Extract metrics if available
    for env in HalfCheetah-v4 Hopper-v4 Walker2d-v4; do
        for algo in DDPG TD3; do
            LOG="$OUT_DIR/ddpg-td3-mujoco-output/${env}/${algo}/train.log"
            if [ -f "$LOG" ]; then
                echo "--- ${algo} ${env} ---"
                tail -3 "$LOG"
            fi
        done
    done

    # Show comparison plots
    echo ""
    echo "--- Comparison PNGs ---"
    ls -la "$OUT_DIR/ddpg-td3-mujoco-output/comparison/" 2>/dev/null || echo "  (none)"

    echo ""
    echo "=== CRON: Training complete. Run 'cron delete' to stop monitoring. ==="
fi

echo "=== CHECK DONE $TIMESTAMP ==="
