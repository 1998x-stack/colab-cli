#!/bin/bash
# Cron watchtower payload — fetch training artifacts from Colab VM.
# Excludes checkpoints and model files (.pt, .pth) — those stay on Drive.
#
# Usage: bash fetch.sh <session_name> [output_dir] [account]
#   session_name: Colab session name (e.g., alphago-1)
#   output_dir: local dir for artifacts (default: ./output)
#   account: colab|cb|cc|clb (default: colab)
#
# Exit codes: 0=OK, 1=session dead, 2=exec failed, 3=download failed

set -euo pipefail

SESSION="${1:?Usage: fetch.sh <session_name> [output_dir] [account]}"
OUTDIR="${2:-projects/alphago/output}"
ACCOUNT="${3:-colab}"

# Resolve account alias to full env
case "$ACCOUNT" in
    colab) COL="colab" ;;
    cb)    COL="cb" ;;
    cc)    COL="cc" ;;
    clb)   COL="clb" ;;
    *)     echo "ERROR: unknown account: $ACCOUNT"; exit 2 ;;
esac

mkdir -p "$OUTDIR/logs" "$OUTDIR/pngs"

# ---- Step 1: Check session alive ----
echo "=== [$(date '+%H:%M:%S')] Fetching from $SESSION (account=$ACCOUNT) ==="

if ! $COL sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[FATAL] Session '$SESSION' is DEAD or not found."
    echo "  Active sessions:"
    $COL sessions 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "  Session alive."

# ---- Step 2: Tar VM output (exclude checkpoints, model files, replay buffer) ----
echo "  Taring VM output..."
TAR_SCRIPT='
import subprocess as sp, sys
cmd = [
    "tar", "-czf", "/content/alphago-output.tar.gz",
    "-C", "/content/alphago-output",
    "--exclude=checkpoints",
    "--exclude=*.pt",
    "--exclude=*.pth",
    "--exclude=replay_buffer",
    "."
]
r = sp.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(f"tar stderr: {r.stderr}", file=sys.stderr)
    sys.exit(1)
print("tar OK")
'

if ! echo "$TAR_SCRIPT" | $COL exec -s "$SESSION" --timeout 15 2>/dev/null; then
    echo "  WARNING: tar via exec failed — trying direct download fallback"
    # Fallback: try downloading individual files directly
fi

# ---- Step 3: Download tar ----
echo "  Downloading..."
if ! $COL download -s "$SESSION" /content/alphago-output.tar.gz "$OUTDIR/output.tar.gz" 2>/dev/null; then
    echo "  WARNING: Download failed — VM may be unreachable or tar not found"
    exit 3
fi

# ---- Step 4: Extract ----
echo "  Extracting..."
tar -xzf "$OUTDIR/output.tar.gz" -C "$OUTDIR/" 2>/dev/null || {
    echo "  WARNING: Extract failed (possibly empty tar)"
}

# ---- Step 5: Report ----

echo ""
echo "=== Log tail (last 15 lines) ==="
if [ -f "$OUTDIR/logs/train.log" ]; then
    tail -15 "$OUTDIR/logs/train.log"
    echo ""
    # Detect FATAL or error
    if grep -q "FATAL" "$OUTDIR/logs/train.log" 2>/dev/null; then
        echo "  FATAL detected in log!"
    fi
else
    echo "  (no log file yet)"
fi

echo ""
echo "=== CSV tail (last 3 rows) ==="
if [ -f "$OUTDIR/metrics.csv" ]; then
    head -1 "$OUTDIR/metrics.csv"
    tail -3 "$OUTDIR/metrics.csv"
else
    echo "  (no metrics CSV yet)"
fi

echo ""
echo "=== Latest PNGs ==="
if [ -d "$OUTDIR/pngs" ] && [ "$(ls -A "$OUTDIR/pngs" 2>/dev/null)" ]; then
    ls -la "$OUTDIR/pngs/"
else
    echo "  (no PNGs yet)"
fi

echo ""
echo "=== Status ==="
if [ -f "$OUTDIR/summary.json" ]; then
    python3 -c "
import json
with open('$OUTDIR/summary.json') as f:
    s = json.load(f)
print(f\"  Iteration: {s.get('iteration', '?')}\")
print(f\"  Policy loss: {s['train_metrics'].get('policy_loss', '?'):.4f}\")
print(f\"  Value loss:  {s['train_metrics'].get('value_loss', '?'):.4f}\")
print(f\"  Win rate:    {s['eval_metrics'].get('win_rate', '?'):.3f}\")
print(f\"  Best model:  {s.get('is_best', False)}\")
print(f\"  Elapsed:     {s.get('elapsed_s', 0):.0f}s\")
print(f\"  Positions:   {s.get('n_positions', 0)}\")
" 2>/dev/null || echo "  (summary parse failed)"
fi

echo ""
echo "=== fetch.sh done ==="
