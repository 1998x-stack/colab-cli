#!/bin/bash
# deploy.sh — Provision Colab, upload notebook + utils, launch papermill.
#
# Usage: bash deploy.sh [cpu|t4]
#   cpu  — CPU session (default, sklearn doesn't need GPU)
#   t4   — GPU session (if you want CUDA-accelerated sklearn, rare)

set -euo pipefail

ACCEL="${1:-cpu}"
SESSION="sklearn-tutorial"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$HOME/Desktop/projects/colab-cli/.claude/skills/colab-cli/scripts"

echo "=== Deploying sklearn tutorial to Colab ==="
echo "Session: $SESSION  |  Accelerator: $ACCEL"

# ---- Proxy (Config B — HTTP CONNECT, most reliable for full workflow) ----
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# ---- Provision ----
echo ""
echo "[1/4] Provisioning Colab session..."
if [ "$ACCEL" = "t4" ]; then
    colab new --gpu T4 -s "$SESSION"
else
    colab new -s "$SESSION"
fi
echo "Session created: $SESSION"

# ---- Upload files ----
echo ""
echo "[2/4] Uploading files..."
colab upload "$PROJ_DIR/tutorial.ipynb" /content/tutorial.ipynb
colab upload "$SKILL_DIR/log_utils.py" /content/log_utils.py
colab upload "$SKILL_DIR/plot_utils.py" /content/plot_utils.py
echo "Files uploaded."

# ---- Launch papermill ----
echo ""
echo "[3/4] Launching papermill..."
cd "$PROJ_DIR"
colab exec -f run_notebook.py --timeout 30
echo "Launcher returned. Papermill should be running detached."

# ---- Quick verify ----
echo ""
echo "[4/4] Quick verify..."
sleep 5
echo 'import subprocess; r=subprocess.run(["pgrep","-f","papermill"],capture_output=True,text=True); print("papermill running" if r.stdout else "papermill NOT running — check log")' | colab exec -s "$SESSION" --timeout 10

echo ""
echo "=== Deploy complete ==="
echo "Session: $SESSION"
echo "Monitor:  echo 'import subprocess; subprocess.run([\"tail\",\"-10\",\"/content/papermill.log\"])' | colab exec -s $SESSION --timeout 10"
echo "Fetch:    bash $PROJ_DIR/fetch.sh $SESSION"
echo "Stop:     colab stop -s $SESSION"
