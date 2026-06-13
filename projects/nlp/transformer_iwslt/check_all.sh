#!/bin/bash
# Quick check on all 3 experiments
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COLAB="$HOME/.local/bin/colab"

check_one() {
    local home="$1" session="$2" label="$3"
    echo "=== $label ($session) ==="
    (
        export HTTPS_PROXY=http://127.0.0.1:7890
        export HTTP_PROXY=http://127.0.0.1:7890
        export ALL_PROXY=socks5://127.0.0.1:7890
        export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
        [ -n "$home" ] && export HOME="$home"
        "$COLAB" exec -s "$session" -f "$SCRIPT_DIR/check_progress.py" --timeout 15 2>/dev/null || \
            echo "[$label] Session dead or check failed"
    )
}

check_one "" transformer-baseline baseline
check_one "$HOME/colab-accounts/account-b" transformer-fixedpe fixed_pe
check_one "$HOME/colab-accounts/account-clb" transformer-heads1 heads_1
