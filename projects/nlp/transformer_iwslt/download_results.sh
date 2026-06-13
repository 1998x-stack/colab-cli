#!/bin/bash
# Download results from all 3 Colab sessions and generate charts.
#
# Usage:
#   ./download_results.sh              # download from all sessions
#   ./download_results.sh baseline     # download baseline only
#   ./download_results.sh --charts     # generate charts only (after downloads)
#
# Uses tar for directories (colab download doesn't do dirs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

download_experiment() {
    local exp="$1"
    local account="${ACCOUNTS[$exp]}"
    local session="${SESSIONS[$exp]}"
    local out="$SCRIPT_DIR/output-${exp}"

    mkdir -p "$out" "$out/checkpoints"

    echo "=== Downloading $exp from $account:$session ==="

    # Download metrics
    echo "[$exp] Downloading metrics.jsonl..."
    $account download /content/metrics.jsonl "$out/metrics.jsonl" 2>/dev/null || \
        echo "[$exp] WARNING: metrics.jsonl not found"

    # Download config
    echo "[$exp] Downloading config.json..."
    $account download /content/config.json "$out/config.json" 2>/dev/null || \
        echo "[$exp] WARNING: config.json not found"

    # Tar + download checkpoints
    echo "[$exp] Taring checkpoints on VM..."
    echo 'import tarfile, os; d="/content/checkpoints"; t=tarfile.open("/content/ckpts.tar.gz","w:gz"); t.add(d,"checkpoints") if os.path.exists(d) else None; t.close()' | \
        $account exec -s "$session" --timeout 30 2>/dev/null || true

    echo "[$exp] Downloading checkpoints..."
    $account download /content/ckpts.tar.gz "$out/checkpoints.tar.gz" 2>/dev/null && \
        (cd "$out" && tar -xzf checkpoints.tar.gz 2>/dev/null && echo "[$exp] Checkpoints extracted") || \
        echo "[$exp] WARNING: checkpoints download failed (may already be gone)"

    # Try individual checkpoint download if tar failed
    if [ ! -f "$out/checkpoints.tar.gz" ]; then
        echo "[$exp] Trying individual checkpoint download..."
        for epoch in 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
            ckpt="checkpoint_epoch${epoch}.pt"
            $account download "/content/checkpoints/$ckpt" "$out/checkpoints/$ckpt" 2>/dev/null && \
                echo "[$exp] Downloaded $ckpt"
        done
    fi

    # Download train log
    echo "[$exp] Downloading train.log..."
    $account download /content/train.log "$out/train.log" 2>/dev/null || \
        echo "[$exp] WARNING: train.log not found"

    echo "[$exp] Done"
    echo ""
}

# ============================================================
# Main
# ============================================================

if [ "${1:-}" = "--charts" ]; then
    echo "=== Generating charts ==="
    cd "$SCRIPT_DIR" && python charts.py
    echo "Charts saved to $SCRIPT_DIR/charts/"
    echo ""
    ls -la "$SCRIPT_DIR/charts/"
    exit 0
fi

if [ $# -gt 0 ] && [ "$1" != "--all" ]; then
    # Download specific experiment
    download_experiment "$1"
else
    # Download all
    for exp in baseline fixed_pe heads_1; do
        download_experiment "$exp"
    done
fi

echo "=== All downloads complete ==="
echo ""
echo "To generate charts: $0 --charts"
echo ""
echo "Output directories:"
for exp in baseline fixed_pe heads_1; do
    out="$SCRIPT_DIR/output-${exp}"
    if [ -d "$out" ]; then
        echo "  $out/ ($(ls "$out" 2>/dev/null | wc -l | tr -d ' ') files)"
    fi
done
