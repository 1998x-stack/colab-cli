"""Benchmark sync-003: CUDA timing without synchronize() is 10-100× wrong.

Without torch.cuda.synchronize(), time.perf_counter() measures CPU submission
latency (~5µs), not GPU execution time (~500µs).
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/sync-003")
WARMUP = 10
ITERS = 50


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("sync-003: CUDA timing without synchronize()")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "method", "avg_us", "error_ratio"])
            csv_w.writeheader()

            matmul_sizes = [64, 128, 256, 512, 1024]

            for n in matmul_sizes:
                a = torch.randn(n, n, device="cuda")
                b = torch.randn(n, n, device="cuda")

                # Warmup
                for _ in range(WARMUP):
                    c = a @ b
                torch.cuda.synchronize()

                # Method A: NO sync — measures CPU submission time (WRONG)
                times_no_sync = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    c = a @ b  # launched asynchronously!
                    t1 = time.perf_counter()  # measures CPU wall clock only
                    times_no_sync.append(t1 - t0)
                t_no_sync = sum(times_no_sync) / len(times_no_sync)

                # Method B: WITH sync — measures actual GPU time (CORRECT)
                times_sync = []
                for _ in range(ITERS):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    c = a @ b
                    torch.cuda.synchronize()  # wait for GPU to finish
                    times_sync.append(time.perf_counter() - t0)
                t_sync = sum(times_sync) / len(times_sync)

                # Method C: CUDA events (gold standard)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                times_event = []
                for _ in range(ITERS):
                    start.record()
                    c = a @ b
                    end.record()
                    torch.cuda.synchronize()
                    times_event.append(start.elapsed_time(end) / 1000)  # ms → s
                t_event = sum(times_event) / len(times_event)

                error_ratio = t_sync / max(t_no_sync, 1e-9)

                log_msg(f"\n  {n}×{n} matmul:")
                log_msg(f"    No sync (CPU submission):  {t_no_sync*1e6:>8.1f} µs  ← WRONG")
                log_msg(f"    With sync (GPU actual):    {t_sync*1e6:>8.1f} µs  ← CORRECT")
                log_msg(f"    CUDA events (gold):        {t_event*1e6:>8.1f} µs")
                log_msg(f"    Error ratio:               {error_ratio:.0f}× (no-sync underestimates by {error_ratio:.0f}×)")

                csv_w.writerow({"size": f"{n}×{n}", "method": "no_sync", "avg_us": round(t_no_sync * 1e6, 1), "error_ratio": round(error_ratio, 1)})
                csv_w.writerow({"size": f"{n}×{n}", "method": "with_sync", "avg_us": round(t_sync * 1e6, 1), "error_ratio": 1.0})
                csv_w.writerow({"size": f"{n}×{n}", "method": "cuda_events", "avg_us": round(t_event * 1e6, 1), "error_ratio": 1.0})

                del a, b, c, start, end

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
