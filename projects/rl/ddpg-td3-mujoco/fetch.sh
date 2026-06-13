#!/bin/bash
# Fetch DDPG vs TD3 MuJoCo training outputs from Colab VM to local project dir.
# Called by cron every 2 minutes. Uses account cb (stefaniehu929).
set -euo pipefail

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/ddpg-td3-mujoco/output"
SESSION="ddpg-td3"

# Account cb: stefaniehu929
export HOME="$HOME/colab-accounts/account-b"
COLAB="/Users/mx/.local/bin/colab"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FETCH $TIMESTAMP (session=$SESSION, account=cb) ==="

# 1. Check session alive
echo "--- Session check ---"
$COLAB sessions 2>&1 | grep "$SESSION" || { echo "SESSION $SESSION NOT FOUND — may be dead"; exit 1; }
echo "OK: session alive"

# 2. Tar output on VM
echo "--- Taring on VM ---"
echo '
import subprocess, os
root = "/content/ddpg-td3-mujoco-output"
if os.path.exists(root):
    subprocess.run(["tar", "-czf", "/content/ddpg-td3-out.tar.gz", "-C", "/content", "ddpg-td3-mujoco-output"], check=True)
    size = os.path.getsize("/content/ddpg-td3-out.tar.gz") / 1024
    print(f"OK tar: {size:.0f} KB")
else:
    print("Output dir not found yet")
' | $COLAB exec -s "$SESSION" --timeout 20 2>&1 || echo "  tar failed (may not exist yet)"

# 3. Download tar
echo "--- Downloading ---"
$COLAB download -s "$SESSION" /content/ddpg-td3-out.tar.gz "/tmp/ddpg-td3-out.tar.gz" 2>&1 || {
    echo "  tar download failed — trying individual logs"
    for env in HalfCheetah-v4 Hopper-v4 Walker2d-v4; do
        for algo in DDPG TD3; do
            mkdir -p "$OUT_DIR/${env}/${algo}"
            $COLAB download -s "$SESSION" \
                "/content/ddpg-td3-mujoco-output/${env}/${algo}/train.log" \
                "$OUT_DIR/${env}/${algo}/train.log" 2>&1 || true
        done
    done
}

# 4. Extract
TAR="/tmp/ddpg-td3-out.tar.gz"
if [ -f "$TAR" ]; then
    mkdir -p "$OUT_DIR"
    tar -xzf "$TAR" -C "$OUT_DIR/" 2>&1 && echo "OK: extracted" || echo "  extract failed"
fi

# 5. Show summary
echo ""
echo "--- Training Status ---"

for env in HalfCheetah-v4 Hopper-v4 Walker2d-v4; do
    for algo in DDPG TD3; do
        # Check both possible paths (tar extract vs direct download)
        LOG1="$OUT_DIR/ddpg-td3-mujoco-output/${env}/${algo}/train.log"
        LOG2="$OUT_DIR/${env}/${algo}/train.log"
        LOG=""
        [ -f "$LOG1" ] && LOG="$LOG1"
        [ -f "$LOG2" ] && LOG="$LOG2"

        if [ -n "$LOG" ] && [ -s "$LOG" ]; then
            LAST=$(tail -1 "$LOG" 2>/dev/null || echo "empty")
            echo "  $algo / $env: $LAST"
        else
            echo "  $algo / $env: no log yet"
        fi
    done
done

echo ""
echo "--- Comparison plots ---"
ls -lt "$OUT_DIR/ddpg-td3-mujoco-output/comparison/"*_comparison.png 2>/dev/null | head -5 || echo "  (none yet)"

echo ""
echo "=== FETCH DONE $TIMESTAMP ==="
