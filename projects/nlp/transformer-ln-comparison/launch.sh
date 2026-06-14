#!/bin/bash
# Deploy Post-LN and Pre-LN training to two Colab accounts in parallel.
#
# Usage:
#   bash launch.sh              # Launch both post-LN (colab) and pre-LN (cb)
#   bash launch.sh --post       # Launch only post-LN on colab account
#   bash launch.sh --pre        # Launch only pre-LN on cb account
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train.py"

# Proxy config — Config B (HTTP CONNECT, full workflow)
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

MODE="${1:-all}"

launch_one() {
    local account="$1"       # colab | cb
    local session="$2"       # transformer-postln | transformer-preln
    local ln_type="$3"       # post | pre
    local home_dir="$4"      # HOME path for the account

    echo "============================================"
    echo " Launching ${ln_type^^}-LN on $account ($session)"
    echo "============================================"

    # Provision GPU session
    echo "[1/3] Provisioning GPU session..."
    HOME="$home_dir" /Users/mx/.local/bin/colab new --gpu T4 -s "$session" 2>/dev/null || true
    sleep 2

    # Verify session exists
    if ! HOME="$home_dir" /Users/mx/.local/bin/colab sessions 2>/dev/null | grep -q "$session"; then
        echo "ERROR: Session $session not found. Check account $account."
        return 1
    fi
    echo "  Session $session ready."

    # Upload train.py
    echo "[2/3] Uploading train.py..."
    HOME="$home_dir" /Users/mx/.local/bin/colab upload "$TRAIN_PY" /content/train.py 2>/dev/null
    echo "  Uploaded."

    # Launch training
    echo "[3/3] Launching training (detached subprocess)..."
    HOME="$home_dir" /Users/mx/.local/bin/colab exec -s "$session" --timeout 120 <<PYEOF
import subprocess, sys, os, time

# Install deps (matplotlib for plotting)
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open("/content/train.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py",
         "--ln_type", "${ln_type}",
         "--max_steps", "500",
         "--batch_size", "64",
         "--log_interval", "25"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

time.sleep(3)
if proc.poll() is not None:
    print(f"ERROR: training exited immediately code={proc.returncode}")
    subprocess.run(["tail", "-20", "/content/train.log"])
    sys.exit(1)

print(f"OK. PID={proc.pid}")
print(f"Monitor: HOME={home_dir} colab exec -s {session} '!tail -f /content/train.log'")
PYEOF

    echo ""
    echo "  ${ln_type^^}-LN launched on $account ($session)"
    echo ""
}

if [ "$MODE" = "all" ] || [ "$MODE" = "--post" ]; then
    launch_one "colab" "transformer-postln" "post" "$HOME"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--pre" ]; then
    launch_one "cb" "transformer-preln" "pre" "$HOME/colab-accounts/account-b"
fi

echo "Done. Monitor with: bash fetch.sh"
