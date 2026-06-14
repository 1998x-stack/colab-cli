#!/bin/bash
# Cron fetch script — monitor both post-LN and pre-LN Colab training runs.
# Designed for 4-minute cron interval.
#
# Usage:
#   bash fetch.sh                    # Fetch both accounts
#   bash fetch.sh --post             # Fetch only post-LN
#   bash fetch.sh --pre              # Fetch only pre-LN
#   bash fetch.sh --summary          # Side-by-side comparison summary
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$SCRIPT_DIR/output"

# Proxy config
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

COLAB_BIN="/Users/mx/.local/bin/colab"
MODE="${1:-all}"

mkdir -p "$LOCAL_OUT/postln/logs" "$LOCAL_OUT/postln/pngs"
mkdir -p "$LOCAL_OUT/preln/logs"  "$LOCAL_OUT/preln/pngs"

# ---------------------------------------------------------------------------
fetch_one() {
    local account="$1"       # colab | cb
    local session="$2"       # transformer-postln | transformer-preln
    local label="$3"         # postln | preln
    local home_dir="$4"      # HOME for account
    local local_dir="$LOCAL_OUT/$label"

    echo ""
    echo "=== [$label] $(date '+%Y-%m-%d %H:%M:%S') ==="

    # 1. Check session alive
    if ! HOME="$home_dir" $COLAB_BIN sessions 2>/dev/null | grep -q "$session"; then
        echo "  [$label] SESSION DEAD — skipping"
        return 1
    fi
    echo "  Session: alive"

    # 2. Tar output on VM (exclude checkpoints to keep tar small)
    local tar_output
    tar_output=$(HOME="$home_dir" $COLAB_BIN exec -s "$session" --timeout 15 2>/dev/null <<'PYEOF'
import subprocess, os
d = "/content/transformer-ln-comparison-output"
if os.path.isdir(d):
    subprocess.run(["tar", "-czf", "/content/fetch.tar.gz",
                    "-C", d, "--exclude=checkpoints", "."], check=True)
    sz = os.path.getsize("/content/fetch.tar.gz")
    print(f"TAR_OK:{sz}")
else:
    print("NO_OUTDIR")
PYEOF
) || true

    # 3. Download
    if echo "$tar_output" | grep -q "TAR_OK"; then
        local tar_size
        tar_size=$(echo "$tar_output" | grep "TAR_OK" | cut -d: -f2)
        echo "  Tar: ${tar_size} bytes on VM"

        HOME="$home_dir" $COLAB_BIN download -s "$session" \
            /content/fetch.tar.gz "$local_dir/fetch.tar.gz" 2>/dev/null || {
            echo "  Download FAILED — falling back to individual files"
            _fallback_download "$home_dir" "$session" "$local_dir"
            return
        }

        # Extract
        tar -xzf "$local_dir/fetch.tar.gz" -C "$local_dir/" 2>/dev/null || true
        echo "  Downloaded + extracted."
    else
        echo "  Tar failed (VM not ready?) — falling back to individual files"
        _fallback_download "$home_dir" "$session" "$local_dir"
        return
    fi

    # 4. Report
    _report "$local_dir" "$label"
}

_fallback_download() {
    local home_dir="$1"
    local session="$2"
    local local_dir="$3"

    HOME="$home_dir" $COLAB_BIN download -s "$session" \
        /content/transformer-ln-comparison-output/logs/train.log \
        "$local_dir/logs/train.log" 2>/dev/null || true
    HOME="$home_dir" $COLAB_BIN download -s "$session" \
        /content/transformer-ln-comparison-output/metrics.csv \
        "$local_dir/metrics.csv" 2>/dev/null || true
    HOME="$home_dir" $COLAB_BIN download -s "$session" \
        /content/transformer-ln-comparison-output/pngs/training_curves.png \
        "$local_dir/pngs/training_curves.png" 2>/dev/null || true
}

_report() {
    local dir="$1"
    local label="$2"

    # Log tail
    if [ -f "$dir/logs/train.log" ]; then
        echo "  --- Log tail (last 8) ---"
        tail -8 "$dir/logs/train.log" | while IFS= read -r line; do
            echo "  $line"
        done
    else
        echo "  (no log yet)"
    fi

    # Metrics
    if [ -f "$dir/metrics.csv" ]; then
        local rows
        rows=$(wc -l < "$dir/metrics.csv")
        echo "  --- Metrics ($((rows - 1)) rows) ---"
        echo -n "  "; head -1 "$dir/metrics.csv"
        echo -n "  "; tail -1 "$dir/metrics.csv"
    fi

    # PNG
    if [ -f "$dir/pngs/training_curves.png" ]; then
        echo "  PNG: $(ls -lh "$dir/pngs/training_curves.png" | awk '{print $5}') $(date -r "$dir/pngs/training_curves.png" '+%H:%M:%S')"
    fi
}

# ---------------------------------------------------------------------------
_show_summary() {
    echo ""
    echo "============================================"
    echo " Post-LN vs Pre-LN Summary @ $(date '+%H:%M:%S')"
    echo "============================================"

    for label in postln preln; do
        local csv="$LOCAL_OUT/$label/metrics.csv"
        if [ -f "$csv" ]; then
            local rows
            rows=$(wc -l < "$csv")
            local last
            last=$(tail -1 "$csv")
            local step
            step=$(echo "$last" | cut -d, -f1)
            local loss
            loss=$(echo "$last" | cut -d, -f2)
            local val_loss
            val_loss=$(echo "$last" | cut -d, -f3)
            local upper_label
            upper_label=$(echo "$label" | tr '[:lower:]' '[:upper:]')
            echo "  ${upper_label}: step=$step | train_loss=$loss | val_loss=$val_loss | rows=$((rows - 1))"
        else
            echo "  ${label^^}: (no data yet)"
        fi
    done

    # Side-by-side comparison if both have data
    local post_csv="$LOCAL_OUT/postln/metrics.csv"
    local pre_csv="$LOCAL_OUT/preln/metrics.csv"
    if [ -f "$post_csv" ] && [ -f "$pre_csv" ]; then
        echo ""
        echo "  --- Loss comparison ---"
        local post_loss
        post_loss=$(tail -1 "$post_csv" | cut -d, -f2)
        local pre_loss
        pre_loss=$(tail -1 "$pre_csv" | cut -d, -f2)
        local post_val
        post_val=$(tail -1 "$post_csv" | cut -d, -f3)
        local pre_val
        pre_val=$(tail -1 "$pre_csv" | cut -d, -f3)

        if [ -n "$post_loss" ] && [ -n "$pre_loss" ]; then
            echo "  Train loss:  Post-LN=$post_loss  Pre-LN=$pre_loss"
            echo "  Val loss:    Post-LN=$post_val  Pre-LN=$pre_val"
            # Simple comparison: which is lower?
            local train_winner val_winner
            train_winner=$(python3 -c "
post='$post_loss'; pre='$pre_loss'
if post == 'nan' or post == '': print('Pre-LN stable (post-LN NaN)')
elif pre == 'nan' or pre == '': print('Post-LN stable (pre-LN NaN)')
else: print('Pre-LN better' if float(pre) < float(post) else 'Post-LN better')
")
            val_winner=$(python3 -c "
post='$post_val'; pre='$pre_val'
if post == 'nan' or post == '': print('Pre-LN stable (post-LN NaN)')
elif pre == 'nan' or pre == '': print('Post-LN stable (pre-LN NaN)')
else: print('Pre-LN better' if float(pre) < float(post) else 'Post-LN better')
")
            echo "  Train winner: $train_winner"
            echo "  Val winner:   $val_winner"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ "$MODE" = "--summary" ]; then
    _show_summary
    exit 0
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--post" ]; then
    fetch_one "colab" "transformer-postln" "postln" "$HOME"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--pre" ]; then
    fetch_one "cb" "transformer-preln" "preln" "$HOME/colab-accounts/account-b"
fi

_show_summary

echo ""
echo "Done. Run 'python3 charts.py' to generate comparison visualization."
