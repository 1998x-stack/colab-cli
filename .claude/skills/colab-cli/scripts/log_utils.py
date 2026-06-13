"""Reusable logging utilities for Colab/Kaggle training scripts.

Usage:
    from log_utils import Logger, MetricsCSV, SummaryJSON

    logger = Logger("/content/output/logs/train.log")
    logger.log("Training started")
    logger.log(f"Ep 1/5 | Batch 100 | loss=1.23 | avg100=1.35")

    csv = MetricsCSV("/content/output/metrics.csv",
                     ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                      "elapsed_s", "lr"])
    csv.write_row(1, 1.234, 0.456, 1.345, 0.500, 180.0, 0.089)

    summary = SummaryJSON("/content/output/summary.json")
    summary.write({"test_acc": 0.87, "epochs_completed": 5, "total_time_s": 900})
"""

import json
import os
import sys
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Logger — timestamped, prints to stdout AND appends to file
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Timestamped log lines to stdout + file. Use flush=True on every print."""

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Tee — redirect stdout to file + terminal (for capturing ALL print output)
# ═══════════════════════════════════════════════════════════════════════════════

class Tee:
    """Duplicate stdout to a file. Call Tee(path) early in main() to capture all prints.

    Usage:
        tee = Tee("/content/output/logs/train.log")
        print("This goes to stdout AND the log file")
    """

    def __init__(self, path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.log_file = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MetricsCSV — header-on-first-write, append rows, crash-safe
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCSV:
    """Structured CSV with header validation. Rows keyed by epoch, write on each epoch end.

    Usage:
        csv = MetricsCSV("/content/output/metrics.csv",
                         ["epoch","train_loss","train_acc","test_loss","test_acc",
                          "elapsed_s","lr"])
        # At end of epoch 1:
        csv.write_row(epoch=1, train_loss=1.23, train_acc=0.45,
                      test_loss=1.34, test_acc=0.50, elapsed_s=180, lr=0.089)
    """

    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write header on creation
        with open(self.path, "w") as f:
            f.write(",".join(self.columns) + "\n")

    def write_row(self, **kwargs):
        """Append a row. Keyword names must match column names (order doesn't matter)."""
        row = []
        for col in self.columns:
            val = kwargs.get(col)
            if val is None:
                row.append("")
            elif isinstance(val, float):
                row.append(f"{val:.6f}")
            elif isinstance(val, int):
                row.append(str(val))
            else:
                row.append(str(val))
        with open(self.path, "a") as f:
            f.write(",".join(row) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SummaryJSON — write-once or overwrite final run metadata
# ═══════════════════════════════════════════════════════════════════════════════

class SummaryJSON:
    """Write final run summary as JSON. Call at training end.

    Usage:
        summary = SummaryJSON("/content/output/summary.json")
        summary.write({
            "test_acc": 0.8742,
            "epochs_completed": 5,
            "total_time_s": 912.3,
            "n_params": 11_343_854,
        })
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def update(self, **kwargs):
        """Merge new keys into existing summary (read-modify-write)."""
        existing = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass
        existing.update(kwargs)
        self.write(existing)


def _json_default(obj):
    """Handle non-serializable types (numpy, torch)."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment detection — auto-detect output dir for Colab / Kaggle / local
# ═══════════════════════════════════════════════════════════════════════════════

def detect_output_dir(project_name, local_root="./output"):
    """Return output directory appropriate for the current environment.

    Colab: /content/<project>-output/
    Kaggle: /kaggle/working/<project>-output/
    Local: <local_root>/<project>-output/
    """
    if os.path.exists("/kaggle/working/"):
        base = f"/kaggle/working/{project_name}-output"
    elif os.path.exists("/content/"):
        base = f"/content/{project_name}-output"
    else:
        base = os.path.join(local_root, f"{project_name}-output")
    return base


def setup_output_dirs(out_dir):
    """Create standard output directory structure: logs/ pngs/ checkpoints/."""
    for sub in ["logs", "pngs", "checkpoints"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
