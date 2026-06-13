#!/bin/bash
# Fetch GCN citation training outputs from Colab VM to local project dir.
# Called by cron every 2 minutes. Uses account cb (stefaniehu929).
set -euo pipefail

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/gnn-citation/output"
SESSION="gnn-citation"

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

# 2. Tar output on VM (exclude checkpoints to save bandwidth)
echo "--- Taring on VM ---"
echo '
import subprocess, os
root = "/content/gnn-citation-output"
if os.path.exists(root):
    # Exclude .pt checkpoint files to keep tar small
    subprocess.run([
        "tar", "-czf", "/content/gnn-out.tar.gz",
        "--exclude=*.pt",
        "-C", "/content", "gnn-citation-output"
    ], check=True)
    size = os.path.getsize("/content/gnn-out.tar.gz") / 1024
    print(f"OK tar: {size:.0f} KB")
else:
    print("Output dir not found yet")
' | $COLAB exec -s "$SESSION" --timeout 20 2>&1 || echo "  tar failed (may not exist yet)"

# 3. Download tar
echo "--- Downloading ---"
$COLAB download -s "$SESSION" /content/gnn-out.tar.gz "/tmp/gnn-out.tar.gz" 2>&1 || {
    echo "  tar download failed — trying individual files"
    for ds in Cora CiteSeer PubMed; do
        mkdir -p "$OUT_DIR/${ds}"
        for f in train.log metrics.csv training_curves.png; do
            $COLAB download -s "$SESSION" \
                "/content/gnn-citation-output/${ds}/${f}" \
                "$OUT_DIR/${ds}/${f}" 2>&1 || true
        done
    done
    # Also try comparison dashboard
    mkdir -p "$OUT_DIR/comparison"
    $COLAB download -s "$SESSION" \
        "/content/gnn-citation-output/comparison/comparison_dashboard.png" \
        "$OUT_DIR/comparison/comparison_dashboard.png" 2>&1 || true
}

# 4. Extract
TAR="/tmp/gnn-out.tar.gz"
if [ -f "$TAR" ]; then
    mkdir -p "$OUT_DIR"
    tar -xzf "$TAR" -C "$OUT_DIR/" 2>&1 && echo "OK: extracted" || echo "  extract failed"
fi

# 5. Show summary
echo ""
echo "--- Training Status ---"

for ds in Cora CiteSeer PubMed; do
    # Check both possible paths (tar extract vs direct download)
    LOG1="$OUT_DIR/gnn-citation-output/${ds}/train.log"
    LOG2="$OUT_DIR/${ds}/train.log"
    LOG=""
    [ -f "$LOG1" ] && LOG="$LOG1"
    [ -f "$LOG2" ] && LOG="$LOG2"

    if [ -n "$LOG" ] && [ -s "$LOG" ]; then
        LAST=$(tail -1 "$LOG" 2>/dev/null || echo "empty")
        echo "  $ds: $LAST"
    else
        echo "  $ds: no log yet"
    fi
done

echo ""
echo "--- Comparison dashboard ---"
DASH1="$OUT_DIR/gnn-citation-output/comparison/comparison_dashboard.png"
DASH2="$OUT_DIR/comparison/comparison_dashboard.png"
ls -lt "$DASH1" 2>/dev/null && echo "  (via tar extract)"
ls -lt "$DASH2" 2>/dev/null && echo "  (via direct download)" || echo "  (none yet)"

echo ""
echo "=== FETCH DONE $TIMESTAMP ==="
