#!/usr/bin/env python3
"""Push a Kaggle kernel, poll until complete, download output.

Usage:
    python push_and_wait.py <project_dir> [--slug <slug>] [--poll <seconds>] [--token-file <n>]

    python push_and_wait.py ./my-experiment
    python push_and_wait.py ./my-experiment --slug my-training --poll 60
    python push_and_wait.py ./my-experiment --token-file 4

The project directory must contain:
    - train.py (or other code_file referenced in kernel-metadata.json)
    - kernel-metadata.json

Requires kaggle CLI installed and authenticated.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def resolve_token(token_file: str | None) -> dict[str, str]:
    """Resolve Kaggle API token. Returns env dict or empty dict if using default."""
    if token_file is None:
        # Use default ~/.kaggle/access_token
        default = Path.home() / ".kaggle" / "access_token"
        if default.exists():
            return {}
        # Try project .kaggle/access_token4 as fallback
        project_token = Path(".kaggle/access_token4")
        if project_token.exists():
            return {"KAGGLE_API_TOKEN": project_token.read_text().strip()}
        print("ERROR: No Kaggle token found. Set KAGGLE_API_TOKEN or save to ~/.kaggle/access_token")
        sys.exit(1)

    # Resolve relative to project root
    token_path = Path(".kaggle") / f"access_token{token_file}"
    if not token_path.exists():
        print(f"ERROR: Token file not found: {token_path}")
        sys.exit(1)

    return {"KAGGLE_API_TOKEN": token_path.read_text().strip()}


def read_metadata(project_dir: str) -> tuple[str, str]:
    """Read kernel-metadata.json, return (owner, slug)."""
    meta_path = Path(project_dir) / "kernel-metadata.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run 'kaggle kernels init -p {project_dir}' first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    kernel_id = meta.get("id", "")
    if "/" not in kernel_id:
        print(f"ERROR: kernel-metadata.json 'id' must be 'owner/slug', got '{kernel_id}'")
        sys.exit(1)

    owner, slug = kernel_id.split("/", 1)
    return owner, slug


def push(project_dir: str, env: dict[str, str]) -> str:
    """Push kernel, return full kernel slug."""
    print(f"Pushing {project_dir}...")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", project_dir],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, **env},
    )

    if result.returncode != 0:
        print(f"PUSH FAILED:\n{result.stderr}")
        sys.exit(1)

    # Parse slug from output
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "code/" in line:
            # Extract from URL like https://www.kaggle.com/code/owner/slug
            slug = line.split("code/")[-1].strip()
            print(f"Kernel: {slug}")
            return slug

    print(f"Push output:\n{output}")
    sys.exit(1)


def wait_for_completion(kernel_slug: str, poll_interval: int, env: dict[str, str]) -> bool:
    """Poll until kernel completes or errors. Returns True on success."""
    print(f"Waiting for {kernel_slug} (polling every {poll_interval}s)...")

    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, **env},
        )

        status_line = result.stdout.strip()
        print(f"  [{time.strftime('%H:%M:%S')}] {status_line}")

        status_lower = status_line.lower()
        if "complete" in status_lower:
            return True
        if "error" in status_lower:
            return False
        # Otherwise still running/pending/queued — keep polling

        time.sleep(poll_interval)


def show_logs(kernel_slug: str, env: dict[str, str]):
    """Print kernel logs."""
    result = subprocess.run(
        ["kaggle", "kernels", "logs", kernel_slug],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, **env},
    )

    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return

    for entry in entries:
        stream = entry.get("stream_name", "stdout")
        data = entry.get("data", "")
        prefix = "STDERR:" if stream == "stderr" else ""
        if prefix:
            print(f"{prefix} {data}", end="")
        else:
            print(data, end="")


def download_output(kernel_slug: str, output_dir: str, env: dict[str, str]):
    """Download kernel output."""
    print(f"Downloading output to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", output_dir],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, **env},
    )
    if result.returncode == 0:
        print("Download complete.")
    else:
        print(f"Download error: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Push Kaggle kernel and wait for completion")
    parser.add_argument("project_dir", help="Directory containing kernel-metadata.json and code_file")
    parser.add_argument("--slug", help="Override kernel slug (reads from metadata by default)")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds (default: 60)")
    parser.add_argument("--token-file", help="Token file number (1-4) from .kaggle/ directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory for downloaded results")
    parser.add_argument("--no-download", action="store_true", help="Don't download output after completion")
    args = parser.parse_args()

    env = resolve_token(args.token_file)
    owner, slug = read_metadata(args.project_dir)
    kernel_slug = f"{owner}/{slug}"

    # Push
    returned_slug = push(args.project_dir, env)
    if returned_slug != kernel_slug:
        kernel_slug = returned_slug
        owner = kernel_slug.split("/")[0]

    # Wait
    success = wait_for_completion(kernel_slug, args.poll, env)

    # Show logs on failure
    if not success:
        print("\n--- KERNEL LOGS ---")
        show_logs(kernel_slug, env)
        print("\nKernel failed. See logs above.")
        sys.exit(1)

    print("Kernel completed successfully!")

    # Download
    if not args.no_download:
        download_output(kernel_slug, args.output, env)

    print(f"Done. View at: https://www.kaggle.com/code/{kernel_slug}")


if __name__ == "__main__":
    main()
