"""DL training tricks dispatcher — runs benchmark scripts sequentially via subprocess.

Usage: python train.py --exp_ids dltrain-003,dltrain-004,dltrain-008
"""

import argparse, subprocess, sys, os, time, json
from pathlib import Path

BASE_DIR = "/content/dl-training-output"
SCRIPT_DIR = "/content"


def run_benchmark(exp_id):
    script = Path(SCRIPT_DIR) / f"benchmark_{exp_id.replace('-','_')}.py"
    if not script.exists():
        print(f"[train] SKIP {exp_id}: script not found at {script}")
        return None

    out_dir = Path(BASE_DIR) / exp_id
    env = os.environ.copy()
    env["OUT_DIR"] = str(out_dir)

    print(f"\n{'='*60}")
    print(f"[train] START {exp_id}  →  {out_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", str(script)],
        env=env, cwd=str(script.parent),
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    # Print captured output
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(f"[train] STDERR ({exp_id}):\n{stderr}")

    ok = proc.returncode == 0
    print(f"[train] {'PASS' if ok else 'FAIL'} {exp_id}  rc={proc.returncode}  elapsed={elapsed:.0f}s")

    # Try to read metrics CSV for summary
    summary = {"exp_id": exp_id, "success": ok, "elapsed_s": round(elapsed, 1), "returncode": proc.returncode}
    csv_path = out_dir / "metrics.csv"
    if csv_path.exists():
        summary["metrics_csv"] = str(csv_path)
    log_path = out_dir / "logs" / "benchmark.log"
    if log_path.exists():
        summary["log"] = str(log_path)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", required=True, help="Comma-separated experiment IDs (e.g. dltrain-003,dltrain-004)")
    args = parser.parse_args()

    exp_ids = [e.strip() for e in args.exp_ids.split(",") if e.strip()]

    print(f"[train] Dispatching {len(exp_ids)} experiments: {exp_ids}")
    print(f"[train] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print(f"[train] START at {time.strftime('%H:%M:%S')}")

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    for i, exp_id in enumerate(exp_ids):
        print(f"\n[train] [{i+1}/{len(exp_ids)}] → {exp_id}")
        result = run_benchmark(exp_id)
        if result:
            results.append(result)

    # Write combined summary
    summary_path = Path(BASE_DIR) / "summary.json"
    summary = {
        "total": len(exp_ids),
        "passed": sum(1 for r in results if r and r["success"]),
        "failed": sum(1 for r in results if r and not r["success"]),
        "skipped": len(exp_ids) - len(results),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[train] DONE. {summary['passed']} pass / {summary['failed']} fail / {summary['skipped']} skip")
    print(f"[train] Summary: {summary_path}")


if __name__ == "__main__":
    main()
