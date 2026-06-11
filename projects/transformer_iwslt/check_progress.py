"""Local cron progress checker for Transformer training.

Reads /content/metrics.jsonl on VM, reports status, flags alerts.
"""
import json, os, subprocess, sys


METRICS_PATH = "/content/metrics.jsonl"
LOG_PATH = "/content/train.log"


def check():
    # 1. Read metrics.jsonl for latest epoch
    try:
        with open(METRICS_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print("[check] WARNING: No metrics.jsonl found — training may not have started")
        return 1

    if not lines:
        print("[check] WARNING: metrics.jsonl is empty — no epochs completed yet")
        proc_alive = _pgrep("train.py")
        print(f"[check] Process alive: {proc_alive}")
        return 0 if proc_alive else 1

    latest = json.loads(lines[-1])
    epoch = latest.get("epoch", 0)
    train_loss = latest.get("train_loss", float("inf"))
    val_loss = latest.get("val_loss", float("inf"))
    bleu = latest.get("bleu", 0.0)
    lr = latest.get("lr", 0.0)
    wall_time = latest.get("wall_time_s", 0)

    # 2. Process alive check
    proc_alive = _pgrep("train.py")

    # 3. Log tail
    try:
        with open(LOG_PATH) as f:
            log_lines = f.readlines()
        tail = "".join(log_lines[-5:]).rstrip()
    except FileNotFoundError:
        tail = "(no log file)"

    # 4. Report
    print(f"[check] Epoch: {epoch}/20 | Train Loss: {train_loss:.3f} | "
          f"Val Loss: {val_loss:.3f} | BLEU: {bleu:.1f} | LR: {lr:.8f} | "
          f"Time: {wall_time/60:.1f}m | Process alive: {proc_alive}")

    # 5. Alerts
    alerts = []
    if not proc_alive and epoch < 20:
        alerts.append("CRITICAL: train.py process dead but training incomplete")
    if train_loss > 8:
        alerts.append("WARNING: Train loss >8 — may be diverging")
    if epoch >= 18:
        alerts.append(f"INFO: Near completion — epoch {epoch}/20. Prepare final download.")
    if epoch >= 20:
        alerts.append("DONE: Training complete (epoch 20). Download results.")

    for a in alerts:
        print(f"[check] {a}")

    # 6. Tail recent log
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
