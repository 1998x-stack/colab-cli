#!/bin/bash
# Deploy Transformer IWSLT experiments across 3 Colab accounts in parallel.
#
# Usage:
#   ./deploy.sh              # fresh start — all 3 experiments from epoch 1
#   ./deploy.sh --resume     # resume from latest checkpoints in output-*/
#
# Accounts:
#   colab → baseline   (hackxie1998@gmail.com)
#   cb    → fixed_pe   (stefaniehu929@gmail.com)
#   clb   → heads_1    (xieminghack@gmail.com)
#
# Each account runs independently with its own checkpoint-resume chain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="projects/transformer_iwslt"
MODE="${1:-fresh}"

# --- Experiment configs ---
declare -A ACCOUNTS=(
    [baseline]=colab
    [fixed_pe]=cb
    [heads_1]=clb
)
declare -A SESSIONS=(
    [baseline]=transformer-baseline
    [fixed_pe]=transformer-fixedpe
    [heads_1]=transformer-heads1
)

FILES_TO_UPLOAD=(
    "$SCRIPT_DIR/model.py"
    "$SCRIPT_DIR/train.py"
    "$SCRIPT_DIR/launch.py"
    "$SCRIPT_DIR/checkpoint.py"
)

# ============================================================
# Step 1: Stop any existing sessions (clean slate for fresh)
# ============================================================
echo "=== Step 1: Stopping existing sessions ==="
for exp in baseline fixed_pe heads_1; do
    account="${ACCOUNTS[$exp]}"
    session="${SESSIONS[$exp]}"
    echo "[$exp] Stopping $session on $account..."
    $account stop -s "$session" 2>/dev/null || echo "[$exp] No existing session (OK)"
done
echo ""

# ============================================================
# Step 2: Provision 3 GPU sessions (in parallel)
# ============================================================
echo "=== Step 2: Provisioning 3 GPU sessions ==="
for exp in baseline fixed_pe heads_1; do
    account="${ACCOUNTS[$exp]}"
    session="${SESSIONS[$exp]}"
    (
        echo "[$exp] Provisioning $session on $account..."
        $account new --gpu T4 -s "$session"
        echo "[$exp] Provisioned OK"
    ) &
done
wait
echo ""

# Verify all sessions are running
echo "=== Verifying sessions ==="
colab sessions 2>/dev/null || true
cb sessions 2>/dev/null || true
clb sessions 2>/dev/null || true
echo ""

# ============================================================
# Step 3: Upload code to all 3 sessions
# ============================================================
echo "=== Step 3: Uploading code ==="
for exp in baseline fixed_pe heads_1; do
    account="${ACCOUNTS[$exp]}"
    session="${SESSIONS[$exp]}"
    (
        echo "[$exp] Uploading to $session..."
        for f in "${FILES_TO_UPLOAD[@]}"; do
            $account upload "$f" "/content/$(basename "$f")"
        done
        # Upload exp_id
        echo "$exp" > /tmp/exp_id_$$.txt
        $account upload /tmp/exp_id_$$.txt /content/exp_id.txt
        rm -f /tmp/exp_id_$$.txt
        echo "[$exp] Upload done"
    ) &
done
wait
echo ""

# ============================================================
# Step 4 (resume mode): Upload checkpoints
# ============================================================
if [ "$MODE" = "--resume" ]; then
    echo "=== Step 4: Resume mode — uploading checkpoints ==="
    for exp in baseline fixed_pe heads_1; do
        account="${ACCOUNTS[$exp]}"
        session="${SESSIONS[$exp]}"
        output_dir="$SCRIPT_DIR/output-${exp}/checkpoints"

        if [ -d "$output_dir" ]; then
            # Find latest checkpoint
            latest=$(ls -1 "$output_dir"/checkpoint_epoch*.pt 2>/dev/null | sort -V | tail -1)
            if [ -n "$latest" ]; then
                epoch=$(basename "$latest" .pt | grep -o '[0-9]*$')
                echo "[$exp] Latest checkpoint: epoch $epoch → $latest"
                $account upload "$latest" "/content/checkpoint_epoch${epoch}.pt"
                echo "/content/checkpoint_epoch${epoch}.pt" > /tmp/resume_$$.txt
                $account upload /tmp/resume_$$.txt /content/resume_path.txt
                rm -f /tmp/resume_$$.txt

                # Also upload existing metrics.jsonl to append
                if [ -f "$SCRIPT_DIR/output-${exp}/metrics.jsonl" ]; then
                    $account upload "$SCRIPT_DIR/output-${exp}/metrics.jsonl" /content/metrics.jsonl
                fi
            else
                echo "[$exp] No checkpoints found in $output_dir — starting fresh"
            fi
        else
            echo "[$exp] No output dir — starting fresh"
        fi
    done
    echo ""
fi

# ============================================================
# Step 5: Launch all 3 experiments (in parallel)
# ============================================================
echo "=== Step 5: Launching experiments ==="
for exp in baseline fixed_pe heads_1; do
    account="${ACCOUNTS[$exp]}"
    session="${SESSIONS[$exp]}"
    (
        echo "[$exp] Launching on $account:$session..."
        $account exec -s "$session" -f "$SCRIPT_DIR/launch.py" --timeout 120
        echo "[$exp] Launched OK"
    ) &
done
wait
echo ""

# ============================================================
# Step 6: Quick verify — check train.py is running on each
# ============================================================
echo "=== Step 6: Verifying training started ==="
sleep 15
for exp in baseline fixed_pe heads_1; do
    account="${ACCOUNTS[$exp]}"
    session="${SESSIONS[$exp]}"
    echo "[$exp] Checking..."
    $account exec -s "$session" -f "$SCRIPT_DIR/check_progress.py" --timeout 15 2>/dev/null || \
        echo "[$exp] WARNING: check_progress failed — may still be installing deps"
done
echo ""

# ============================================================
# Step 7: Cron monitoring instructions
# ============================================================
echo "=== Step 7: Cron monitoring ==="
echo ""
echo "Run these in Claude Code to set up monitoring:"
echo ""
echo "CronCreate cron=\"*/5 * * * *\" prompt=\"Check baseline: colab exec -s transformer-baseline -f $SCRIPT_DIR/check_progress.py --timeout 15\" durable=true recurring=true"
echo "CronCreate cron=\"*/5 * * * *\" prompt=\"Check fixed_pe: cb exec -s transformer-fixedpe -f $SCRIPT_DIR/check_progress.py --timeout 15\" durable=true recurring=true"
echo "CronCreate cron=\"*/5 * * * *\" prompt=\"Check heads_1: clb exec -s transformer-heads1 -f $SCRIPT_DIR/check_progress.py --timeout 15\" durable=true recurring=true"
echo ""
echo "=== Deployment complete ==="
echo ""
echo "Session summary:"
echo "  colab → transformer-baseline (baseline)"
echo "  cb    → transformer-fixedpe  (fixed_pe)"
echo "  clb   → transformer-heads1   (heads_1)"
echo ""
echo "To check progress manually:"
echo "  colab exec -s transformer-baseline -f $SCRIPT_DIR/check_progress.py --timeout 15"
echo ""
echo "When session dies (detected by cron):"
echo "  1. Download checkpoint: colab download /content/checkpoints/checkpoint_epoch{N}.pt $SCRIPT_DIR/output-baseline/checkpoints/"
echo "  2. Download metrics:    colab download /content/metrics.jsonl $SCRIPT_DIR/output-baseline/"
echo "  3. Re-run: ./deploy.sh --resume"
