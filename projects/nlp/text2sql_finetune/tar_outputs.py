"""Create tarball of output directory. Survives missing output dirs.

Upload once: colab upload tar_outputs.py /content/text2sql_finetune/tar_outputs.py
Used by fetch.sh for cron-based monitoring.
"""
import subprocess, sys, os

OUTPUT_DIR = os.environ.get("T2S_OUTPUT_DIR", "/content/text2sql-finetune-output")
TAR_PATH = os.environ.get("T2S_TAR_PATH", "/content/text2sql-finetune-output.tar.gz")
TAR_NAME = os.path.basename(OUTPUT_DIR)

if not os.path.isdir(OUTPUT_DIR):
    print(f"ERROR: output dir not found: {OUTPUT_DIR}", file=sys.stderr)
    # Try creating it — maybe training hasn't started yet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created empty {OUTPUT_DIR}")

# Check if there's anything to tar
contents = os.listdir(OUTPUT_DIR)
if not contents:
    print(f"WARNING: {OUTPUT_DIR} is empty — nothing to tar")
    # Create empty tar so download doesn't fail
    subprocess.run(
        ["tar", "-czf", TAR_PATH, "--files-from", "/dev/null"],
        capture_output=True,
    )
    print(f"Created empty {TAR_PATH}")
else:
    result = subprocess.run(
        ["tar", "-czf", TAR_PATH, "-C", os.path.dirname(OUTPUT_DIR), TAR_NAME],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"tar failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    size = os.path.getsize(TAR_PATH)
    print(f"Created {TAR_PATH} ({size} bytes, {len(contents)} entries)")
