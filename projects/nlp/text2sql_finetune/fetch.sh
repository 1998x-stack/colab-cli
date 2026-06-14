#!/bin/bash
# Cron watchtower payload for text2sql_finetune.
# Fires every 2 min from CronCreate. Stops when eval_report.json appears.
#
# Usage: SESSION=<name> bash fetch.sh
set -euo pipefail

SESSION="${SESSION:?must set SESSION env var}"
OUTPUT_DIR="/tmp/text2sql-output-$$"
mkdir -p "$OUTPUT_DIR"

# 1. Check session alive
echo "=== Checking session $SESSION ==="
if ! colab sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "FATAL: session $SESSION not found — stopping cron"
    exit 1
fi

# 2. Tar outputs on VM
echo "=== Tarring outputs on VM ==="
colab exec -s "$SESSION" -f tar_outputs.py --timeout 15 || {
    echo "WARNING: tar failed, trying individual file download"
    colab download -s "$SESSION" /content/text2sql-finetune-output/eval_report.json "$OUTPUT_DIR/eval_report.json" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/train.log "$OUTPUT_DIR/train.log" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql-finetune-output/logs/metrics.csv "$OUTPUT_DIR/metrics.csv" 2>/dev/null || true
}

# 3. Download tarball (if tar succeeded)
if [ -f "output.tar.gz" ] || colab download -s "$SESSION" /content/text2sql-finetune-output.tar.gz "$OUTPUT_DIR/output.tar.gz" 2>/dev/null; then
    if [ -f "output.tar.gz" ]; then
        mv output.tar.gz "$OUTPUT_DIR/output.tar.gz"
    fi
    tar -xzf "$OUTPUT_DIR/output.tar.gz" -C "$OUTPUT_DIR" 2>/dev/null || true
fi

# 4. Report
echo ""
echo "=== train.log (last 5 lines) ==="
tail -5 "$OUTPUT_DIR/logs/train.log" 2>/dev/null || echo "(no train.log yet)"

echo ""
echo "=== metrics.csv (last 3 lines) ==="
tail -3 "$OUTPUT_DIR/metrics.csv" 2>/dev/null || echo "(no metrics.csv yet)"

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
    echo "DONE — eval complete. Remove cron job."
else
    echo "(no eval report yet — training in progress)"
fi

rm -rf "$OUTPUT_DIR"
