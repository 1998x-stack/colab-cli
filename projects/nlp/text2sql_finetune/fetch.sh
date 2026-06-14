#!/bin/bash
# Cron watchtower payload for text2sql_finetune.
# Fires every 2 min from CronCreate.
#
# Usage: SESSION=<name> bash fetch.sh
#
# Resilient to: missing session, WebSocket drops, empty output dir.
# Reports training progress + eval results. Exits 0 even on partial failure
# (cron continues), exits 1 only when session is confirmed dead.
set -euo pipefail

SESSION="${SESSION:?must set SESSION env var}"
OUTPUT_DIR="/tmp/text2sql-output-$$"
mkdir -p "$OUTPUT_DIR"

# ── 1. Check session alive ───────────────────────────────
echo "=== Checking session $SESSION ==="
if ! colab sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "WARNING: session $SESSION not found — may have been reclaimed"
    echo "Training artifacts from the last successful fetch are in /tmp/text2sql-output-*"
    echo "Re-provision with: colab new --gpu T4 -s $SESSION"
    exit 0  # Don't kill cron — user may re-provision
fi

# ── 2. Check training PID on VM ──────────────────────────
echo "=== Training status ==="
PID_STATUS=$(echo 'import os; pf="/content/text2sql-finetune-output/train.pid"
if os.path.exists(pf):
    with open(pf) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        print(f"ALIVE(PID={pid})")
    except OSError:
        print("DEAD")
else:
    print("NO_PID_FILE")' | colab exec -s "$SESSION" --timeout 15 2>/dev/null || echo "WS_FAILED")
echo "  PID: $PID_STATUS"

# ── 3. Tar outputs on VM ─────────────────────────────────
echo "=== Fetching outputs ==="
TAR_OK=false
if colab exec -s "$SESSION" -f tar_outputs.py --timeout 15 2>/dev/null | grep -q "Created"; then
    TAR_OK=true
fi

if $TAR_OK; then
    # Download tarball
    if colab download -s "$SESSION" /content/text2sql-finetune-output.tar.gz "$OUTPUT_DIR/output.tar.gz" 2>/dev/null; then
        tar -xzf "$OUTPUT_DIR/output.tar.gz" -C "$OUTPUT_DIR" 2>/dev/null || true
        echo "  Tarball downloaded + extracted"
    fi
else
    # Fallback: download individual files via REST (survives WebSocket drops)
    echo "  Tar failed — downloading individual files via REST"
    colab download -s "$SESSION" /content/text2sql-finetune-output/eval_report.json "$OUTPUT_DIR/eval_report.json" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/train.log "$OUTPUT_DIR/train.log" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/metrics.csv "$OUTPUT_DIR/metrics.csv" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/eval.log "$OUTPUT_DIR/eval.log" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/watchdog.log "$OUTPUT_DIR/watchdog.log" 2>/dev/null || true
fi

# ── 4. Report ────────────────────────────────────────────
echo ""
echo "=== train.log (last 5 lines) ==="
tail -5 "$OUTPUT_DIR/logs/train.log" 2>/dev/null || echo "(no train.log yet)"

echo ""
echo "=== metrics.csv (last 3 lines) ==="
tail -3 "$OUTPUT_DIR/metrics.csv" 2>/dev/null || echo "(no metrics.csv yet)"

echo ""
echo "=== Watchdog (last 3 lines) ==="
tail -3 "$OUTPUT_DIR/watchdog.log" 2>/dev/null || echo "(no watchdog.log)"

echo ""
echo "=== Eval report ==="
if [ -f "$OUTPUT_DIR/eval_report.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/eval_report.json') as f:
    r = json.load(f)
print(f\"exec_acc={r['execution_accuracy']:.3f}  exact_match={r['exact_match_accuracy']:.3f}  total={r['total']}\")
print(f\"errors: {r['errors']}\")
"
    echo ""
    echo "DONE — eval complete. Training finished successfully."
    echo "Remove cron job and download results:"
    echo "  colab download -s $SESSION /content/text2sql-finetune-output.tar.gz ./output.tar.gz"
else
    echo "(no eval report yet — training in progress)"
    # Estimate progress from metrics
    if [ -f "$OUTPUT_DIR/metrics.csv" ]; then
        STEPS=$(tail -1 "$OUTPUT_DIR/metrics.csv" 2>/dev/null | cut -d',' -f1)
        echo "  Steps completed: ${STEPS:-0}"
    fi
fi

# Cleanup
rm -rf "$OUTPUT_DIR"
