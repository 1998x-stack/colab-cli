"""Local cron progress checker for Transformer-KV-Cache training.

Reads /content/transformer-kv-cache-output/metrics.csv on VM, reports status.
"""
import csv
import subprocess
import sys

METRICS_PATH = "/content/transformer-kv-cache-output/metrics.csv"
LOG_PATH = "/content/train.log"


def check():
    metrics = []
    try:
        with open(METRICS_PATH) as f:
            reader = csv.DictReader(f)
            metrics = list(reader)
    except FileNotFoundError:
        print("[check] WARNING: No metrics.csv found — training may not have started")
        proc_alive = _pgrep("train.py")
        print(f"[check] Process alive: {proc_alive}")
        return 0 if proc_alive else 1

    if not metrics:
        print("[check] WARNING: metrics.csv is empty — no epochs completed")
        print(f"[check] Process alive: {_pgrep('train.py')}")
        return 0

    latest = metrics[-1]
    epoch = int(latest["epoch"])
    train_loss = float(latest["train_loss"])
    val_loss = float(latest["val_loss"])
    ppl = float(latest["perplexity"])
    elapsed = float(latest["elapsed_s"])
    tokens_per_sec = float(latest["tokens_per_sec"])

    proc_alive = _pgrep("train.py")

    try:
        with open(LOG_PATH) as f:
            log_lines = f.readlines()
        tail = "".join(log_lines[-5:]).rstrip()
    except FileNotFoundError:
        tail = "(no log)"

    print(f"[check] Epoch: {epoch} | Train Loss: {train_loss:.3f} | "
          f"Val Loss: {val_loss:.3f} | PPL: {ppl:.1f} | "
          f"tok/s: {tokens_per_sec:.0f} | Time: {elapsed/60:.1f}m | "
          f"Alive: {proc_alive}")

    alerts = []
    if not proc_alive and epoch < 10:
        alerts.append("CRITICAL: train.py dead before epoch 10")
    if train_loss > 6:
        alerts.append("WARNING: Train loss >6 — may be diverging")
    if epoch >= 8:
        alerts.append(f"INFO: Near completion — epoch {epoch}/10.")
    if epoch >= 10:
        alerts.append("DONE: Training complete. Download results.")

    for a in alerts:
        print(f"[check] {a}")

    if tail:
        print(f"[check] Log tail:\n{tail}")

    return 0 if not any("CRITICAL" in a for a in alerts) else 1


def _pgrep(pattern: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(check())
