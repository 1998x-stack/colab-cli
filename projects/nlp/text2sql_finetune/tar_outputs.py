"""Uploaded to Colab VM. tar_outputs.py creates a tarball of the output directory."""
import subprocess
import sys

OUTPUT_DIR = "/content/text2sql-finetune-output"
TAR_PATH = "/content/text2sql-finetune-output.tar.gz"

result = subprocess.run(
    ["tar", "-czf", TAR_PATH, "-C", "/content", "text2sql-finetune-output"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"tar failed: {result.stderr}", file=sys.stderr)
    sys.exit(1)
print(f"Created {TAR_PATH}")
