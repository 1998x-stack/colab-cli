#!/bin/bash
# One-shot deploy text2sql_finetune → Colab GPU with session-survival strategy.
#
# Usage:
#   bash deploy.sh [session_name] [account_suffix]
#
#   bash deploy.sh text2sql          # default account (hackxie1998)
#   bash deploy.sh text2sql cb       # account-b (stefaniehu929)
#
# Session death root cause:
#   KeepAliveAssignment RPC is broken (IAM deadlock, dies at T+61s).
#   WebSocket through Colab runtime proxy is the ONLY liveness signal.
#   Session dies 2-5 min after last WebSocket closes. Any gap is fatal.
#
# Strategy:
#   1. bg_launch.py returns fast — spawns training detached (survives WS drops)
#   2. eval_and_watch.py launched IMMEDIATELY after — keeps WS alive continuously
#   3. Redundant launch (2 attempts, 30s apart) for ~84% success rate from China
#   4. eval_and_watch waits for training, then runs eval with heartbeat every 15s
set -e

SESSION="${1:-text2sql}"
ACCOUNT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Account prefix ───────────────────────────────────────
if [ -n "$ACCOUNT" ]; then
    ACCT_HOME="$HOME/colab-accounts/account-$ACCOUNT"
    COLAB="env HOME=$ACCT_HOME colab"
else
    COLAB="colab"
fi

# ── Proxy: Config B (HTTP CONNECT tunnel, most reliable) ─
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

echo "=== text2sql_finetune → Colab ==="
echo "Session: $SESSION  Account: ${ACCOUNT:-default}"
echo ""

# ── 1. Provision GPU ─────────────────────────────────────
echo "[1/5] Creating GPU session..."
$COLAB new --gpu T4 -s "$SESSION"
echo ""

# ── 2. Create dirs ───────────────────────────────────────
echo "[2/5] Creating directories..."
echo 'import os; os.makedirs("/content/text2sql_finetune/data", exist_ok=True); os.makedirs("/content/text2sql_finetune/logs", exist_ok=True); os.makedirs("/content/text2sql-finetune-output/logs", exist_ok=True); print("ok")' | $COLAB exec -s "$SESSION" --timeout 15
echo ""

# ── 3. Upload source files ───────────────────────────────
echo "[3/5] Uploading source files..."
for f in dataset.py train.py evaluate.py bg_launch.py eval_and_watch.py watchdog.py tar_outputs.py; do
    $COLAB upload "$SCRIPT_DIR/$f" "/content/text2sql_finetune/$f"
    echo "  -> /content/text2sql_finetune/$f"
done
echo ""

# ── 4. Launch detached training ──────────────────────────
echo "[4/5] Launching bg_launch.py (pip install + dataset + spawn training)..."
$COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/bg_launch.py" --timeout 120 2>&1 || {
    echo "NOTE: If TimeoutError, retry: $COLAB exec -s $SESSION -f $SCRIPT_DIR/bg_launch.py --timeout 120"
}
echo ""

# ── 5. Launch eval+watchdog (redundant: 2 attempts) ──────
echo "[5/5] Launching eval_and_watch.py (keeps WS alive, waits for training, runs eval)..."
echo "  Primary + backup (84% success rate from China):"
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/eval_and_watch.py" --timeout 600 \
    > /tmp/text2sql-eval-wd.log 2>&1 &
echo "  Primary PID=$!"
sleep 30
nohup $COLAB exec -s "$SESSION" -f "$SCRIPT_DIR/eval_and_watch.py" --timeout 600 \
    > /tmp/text2sql-eval-wd-backup.log 2>&1 &
echo "  Backup PID=$!"
echo ""

echo "=== Done ==="
echo "Monitor:  tail -f /tmp/text2sql-eval-wd.log"
echo "Download: $COLAB download -s $SESSION /content/text2sql-finetune-output.tar.gz ./output.tar.gz"
echo "Stop:     $COLAB stop -s $SESSION"
