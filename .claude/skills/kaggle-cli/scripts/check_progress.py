#!/usr/bin/env python3
"""Quick health check for a running Kaggle kernel.

Usage:
    python check_progress.py <owner>/<slug> [--token-file <n>] [--tail <lines>]
    python check_progress.py xieming1998/my-training
    python check_progress.py xieming1998/my-training --token-file 4 --tail 20
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def resolve_token(token_file: str | None) -> dict[str, str]:
    if token_file is None:
        default = Path.home() / ".kaggle" / "access_token"
        if default.exists():
            return {}
        project_token = Path(".kaggle/access_token4")
        if project_token.exists():
            return {"KAGGLE_API_TOKEN": project_token.read_text().strip()}
        print("ERROR: No token found.")
        sys.exit(1)

    token_path = Path(".kaggle") / f"access_token{token_file}"
    if not token_path.exists():
        print(f"ERROR: Token file not found: {token_path}")
        sys.exit(1)
    return {"KAGGLE_API_TOKEN": token_path.read_text().strip()}


def main():
    parser = argparse.ArgumentParser(description="Check Kaggle kernel progress")
    parser.add_argument("kernel", help="Kernel slug (owner/slug)")
    parser.add_argument("--token-file", help="Token file number (1-4)")
    parser.add_argument("--tail", type=int, default=0, help="Show last N log lines")
    args = parser.parse_args()

    env = resolve_token(args.token_file)

    # Status
    result = subprocess.run(
        ["kaggle", "kernels", "status", args.kernel],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, **env},
    )
    status = result.stdout.strip()
    print(f"Status: {status}")
    print(f"URL:    https://www.kaggle.com/code/{args.kernel}")

    # Logs
    if args.tail > 0:
        result = subprocess.run(
            ["kaggle", "kernels", "logs", args.kernel],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, **env},
        )
        try:
            entries = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"\nLogs (raw):\n{result.stdout[-2000:]}")
            return

        # Get last N stdout entries
        stdout_entries = [e for e in entries if e.get("stream_name") == "stdout"]
        for entry in stdout_entries[-args.tail:]:
            print(entry["data"], end="")

        # Show stderr last few if they exist
        stderr_entries = [e for e in entries if e.get("stream_name") == "stderr"]
        if stderr_entries:
            print(f"\n--- Last stderr ({len(stderr_entries)} total) ---")
            for entry in stderr_entries[-5:]:
                print(f"STDERR: {entry['data']}", end="")


if __name__ == "__main__":
    main()
