#!/bin/bash
# Deploy Transformer IWSLT experiments across 3 Colab accounts in parallel.
#
# Usage:
#   ./deploy.sh              # fresh start — all 3 experiments from epoch 1
#   ./deploy.sh --resume     # resume from latest checkpoints in output-*/
#
# Accounts (3 free GPU slots, 1 per account):
#   colab → baseline   (hackxie1998@gmail.com)
#   cb    → fixed_pe    (stefaniehu929@gmail.com)
#   clb   → heads_1     (xieminghack@gmail.com)

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COLAB="$HOME/.local/bin/colab"
HF_TOKEN="$SCRIPT_DIR/../../.huggingface/access_token"
MODE="${1:-fresh}"

# ---- helper: run a colab command for a specific account ----
# Usage: colab_for HOME_OVERRIDE session_name command args...
colab_for() {
    local home="$1" session="$2"; shift 2
    (
        export HTTPS_PROXY=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export ALL_PROXY=socks5://127.0.0.1:7890
        export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
        [ -n "$home" ] && export HOME="$home"
        "$COLAB" "$@" -s "$session" 2>/dev/null
    )
}

# ---- helper: provision a session ----
provision() {
    local home="$1" session="$2"
    (
        export HTTPS_PROXY=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export ALL_PROXY=socks5://127.0.0.1:7890
        [ -n "$home" ] && export HOME="$home"
        "$COLAB" stop -s "$session" 2>/dev/null || true
        sleep 1
        "$COLAB" new --gpu T4 -s "$session"
    )
}

# ---- helper: upload code + token, then launch ----
deploy() {
    local home="$1" session="$2" exp_id="$3"
    (
        export HTTPS_PROXY=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export ALL_PROXY=socks5://127.0.0.1:7890
        export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
        [ -n "$home" ] && export HOME="$home"

        for f in model.py train.py launch.py checkpoint.py; do
            "$COLAB" upload "$SCRIPT_DIR/$f" "/content/$f"
        done
        [ -f "$HF_TOKEN" ] && "$COLAB" upload "$HF_TOKEN" /content/hf_token
        echo "$exp_id" > "/tmp/exp_${exp_id}.txt"
        "$COLAB" upload "/tmp/exp_${exp_id}.txt" /content/exp_id.txt
        rm -f "/tmp/exp_${exp_id}.txt"

        # Resume checkpoint if present
        if [ "$MODE" = "--resume" ]; then
            local ckpt_dir="$SCRIPT_DIR/output-${exp_id}/checkpoints"
            if [ -d "$ckpt_dir" ]; then
                local latest=$(ls -1t "$ckpt_dir"/checkpoint_epoch*.pt 2>/dev/null | head -1)
                if [ -n "$latest" ]; then
                    local epoch=$(echo "$latest" | grep -o '[0-9]*\.pt$' | sed 's/\.pt//')
                    echo "[$exp_id] Resume from epoch $epoch"
                    "$COLAB" upload "$latest" "/content/checkpoint_epoch${epoch}.pt"
                    echo "/content/checkpoint_epoch${epoch}.pt" > "/tmp/resume_${exp_id}.txt"
                    "$COLAB" upload "/tmp/resume_${exp_id}.txt" /content/resume_path.txt
                    rm -f "/tmp/resume_${exp_id}.txt"
                    # Upload existing metrics to append
                    local metrics_file="$SCRIPT_DIR/output-${exp_id}/metrics.jsonl"
                    [ -f "$metrics_file" ] && "$COLAB" upload "$metrics_file" /content/metrics.jsonl
                fi
            fi
        fi

        "$COLAB" exec -s "$session" -f "$SCRIPT_DIR/launch.py" --timeout 120
        echo "[$exp_id] Launched"
    )
}

# ============================================================
# Main
# ============================================================

echo "=== Provisioning 3 GPU sessions ==="
provision "" transformer-baseline &
provision "$HOME/colab-accounts/account-b" transformer-fixedpe &
provision "$HOME/colab-accounts/account-clb" transformer-heads1 &
wait
echo ""

echo "=== Deploying code + launching ==="
deploy "" transformer-baseline baseline &
deploy "$HOME/colab-accounts/account-b" transformer-fixedpe fixed_pe &
deploy "$HOME/colab-accounts/account-clb" transformer-heads1 heads_1 &
wait
echo ""

echo "=== All 3 launched ==="
echo "  colab → transformer-baseline (baseline)"
echo "  cb    → transformer-fixedpe  (fixed_pe)"
echo "  clb   → transformer-heads1   (heads_1)"
echo ""
echo "Check progress:"
echo "  $SCRIPT_DIR/check_progress.sh"
echo ""
echo "Download results:"
echo "  $SCRIPT_DIR/download_results.sh"
