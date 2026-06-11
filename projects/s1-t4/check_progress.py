"""Colab progress checker for s1-t4 QLoRA training.

Reads /content/s1-t4/heartbeat.json on VM, reports status, tails logs.
Intended to be run every 5-7 min via CronCreate with colab exec.

Usage (dry-run):
    import check_progress
    check_progress.VM_DIR = "/tmp/test_s1t4"
    check_progress.main()
"""

import json, os, sys, time, glob

# Base directory on the VM — override for dry-run testing
VM_DIR = "/content/s1-t4"


def main() -> int:
    heartbeat_path = os.path.join(VM_DIR, "heartbeat.json")
    log_dir = os.path.join(VM_DIR, "logs")
    checkpoint_dir = os.path.join(VM_DIR, "checkpoints")
    results_dir = os.path.join(VM_DIR, "results")

    # 1. Read heartbeat
    hb = None
    try:
        with open(heartbeat_path) as f:
            hb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[check] WARNING: No heartbeat found at {heartbeat_path}")
        print(f"[check] VM_DIR={VM_DIR}")
        return 1

    now = time.time()
    hb_age = now - hb.get("timestamp", 0)

    status = hb.get("status", "unknown")
    step = hb.get("step", 0)
    loss = hb.get("loss", None)
    elapsed = hb.get("elapsed_seconds", hb_age)

    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
    print(f"[check] Status: {status} | Step: {step} | Loss: {loss_str} | "
          f"Elapsed: {elapsed/60:.1f}m | HB age: {hb_age:.0f}s")

    # 2. Tail train.log (last 15 lines)
    train_log = os.path.join(log_dir, "train.log")
    try:
        with open(train_log) as f:
            lines = f.readlines()
        tail = "".join(lines[-15:]).rstrip()
        if tail:
            print(f"[check] Train log tail ({len(lines)} lines total):")
            print(tail)
    except FileNotFoundError:
        print(f"[check] (train.log not found)")

    # 3. Tail eval.log (last 10 lines, if exists)
    eval_log = os.path.join(log_dir, "eval.log")
    try:
        with open(eval_log) as f:
            lines = f.readlines()
        tail = "".join(lines[-10:]).rstrip()
        if tail:
            print(f"[check] Eval log tail ({len(lines)} lines total):")
            print(tail)
    except FileNotFoundError:
        pass  # eval.log is optional — no message

    # 4. List checkpoints
    ckpt_pattern = os.path.join(checkpoint_dir, "*")
    ckpts = sorted(glob.glob(ckpt_pattern))
    if ckpts:
        # Filter to directories (HF PEFT adapter checkpoints are dirs)
        ckpt_dirs = [c for c in ckpts if os.path.isdir(c)]
        print(f"[check] Checkpoints: {len(ckpt_dirs)} dirs, latest: {os.path.basename(ckpt_dirs[-1])}")
        other = [c for c in ckpts if not os.path.isdir(c)]
        if other:
            print(f"[check]   + {len(other)} files: {[os.path.basename(c) for c in other]}")
    else:
        print(f"[check] Checkpoints: none yet")

    # 5. List results
    results = sorted(glob.glob(os.path.join(results_dir, "*")))
    if results:
        print(f"[check] Results ({len(results)} files):")
        for r in results:
            size = os.path.getsize(r) if os.path.isfile(r) else 0
            label = os.path.basename(r)
            if size:
                print(f"    {label} ({size:,} bytes)")
            else:
                print(f"    {label}")
    else:
        print(f"[check] Results: none yet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
