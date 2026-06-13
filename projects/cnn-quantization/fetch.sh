#!/bin/bash
# Fetch CNN quantization training artifacts from Colab VM via REST.
# Usage: bash fetch.sh <session_name> [proxy_config]
#   proxy_config: "A" for SOCKS5+no_proxy, "B" for HTTP CONNECT (default: B)
set -euo pipefail

SESSION="${1:-cnn-quant}"
PROXY="${2:-B}"
PROJECT="cnn-quantization"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/output"
TAR_NAME="${PROJECT}-output.tar.gz"
REMOTE_TAR="/content/${TAR_NAME}"
LOCAL_TAR="/tmp/${PROJECT}-latest.tar.gz"
REMOTE_DIR="/content/${PROJECT}-output"

# Proxy config
if [ "$PROXY" = "A" ]; then
    export HTTPS_PROXY=socks5://127.0.0.1:7890
    export HTTP_PROXY=socks5://127.0.0.1:7890
    export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
else
    export HTTPS_PROXY=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    export ALL_PROXY=socks5://127.0.0.1:7890
fi

COLAD="$(which colab)"

echo "=== [$(date '+%H:%M:%S')] Fetching ${PROJECT} from session ${SESSION} ==="

# 1. Session check
if ! $COLAD sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "SESSION DEAD: $SESSION not found"
    exit 1
fi
echo "Session alive: $SESSION"

# 2. Tar on VM (exclude checkpoints)
echo "Taring ${REMOTE_DIR}..."
echo "
import subprocess, os
out = '${REMOTE_TAR}'
src = '${REMOTE_DIR}'
# Exclude .pt checkpoints, include everything else
subprocess.run(['tar', '-czf', out, '-C', os.path.dirname(src),
    '--exclude=*.pt', os.path.basename(src)], check=True, timeout=30)
sz = os.path.getsize(out) / 1024
print(f'TAR OK: {sz:.0f} KB')
" | $COLAD exec -s "$SESSION" --timeout 30 2>&1 || {
    echo "TAR FAILED — trying direct download of known paths"
}

# 3. Download tar
echo "Downloading..."
$COLAD download -s "$SESSION" "$REMOTE_TAR" "$LOCAL_TAR" 2>&1 || {
    echo "DOWNLOAD FAILED"
    exit 1
}
ls -lh "$LOCAL_TAR"

# 4. Extract
mkdir -p "$LOCAL_DIR"
tar -xzf "$LOCAL_TAR" --strip-components=1 -C "$LOCAL_DIR" 2>&1
echo "Extracted to $LOCAL_DIR"

# 5. Report
echo ""
echo "=== TAIL -8 logs/train.log ==="
tail -8 "$LOCAL_DIR/logs/train.log" 2>/dev/null || echo "  No log yet"

echo ""
echo "=== TAIL -3 metrics.csv ==="
tail -3 "$LOCAL_DIR/metrics.csv" 2>/dev/null || echo "  No metrics yet"

echo ""
echo "=== QUANTIZATION SUMMARY ==="
cat "$LOCAL_DIR/quantization_summary.csv" 2>/dev/null || echo "  Not yet — training still in progress"

echo ""
echo "=== PNGs ==="
ls -lh "$LOCAL_DIR/pngs/" 2>/dev/null || echo "  No PNGs yet"

echo ""
echo "=== [$(date '+%H:%M:%S')] Fetch complete ==="
