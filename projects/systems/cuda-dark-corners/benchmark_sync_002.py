"""Benchmark sync-002: torch.where vs boolean masking — hidden sync from dynamic shapes.

Boolean masking forces CPU sync to determine output size. torch.where produces
fixed-size output — no sync. torch.where can be 2-5× faster for sparse masks.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/sync-002")
WARMUP = 20
ITERS = 100


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def measure(fn):
    for _ in range(WARMUP):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("sync-002: torch.where vs boolean masking")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["n_elements", "sparsity_pct", "method", "avg_us", "speedup"])
            csv_w.writeheader()

            sizes = [1000, 10000, 100000, 1000000]
            sparsities = [1, 10, 50, 90]

            for n in sizes:
                for sp in sparsities:
                    x = torch.randn(n, device="cuda")
                    mask = torch.rand(n, device="cuda") < (sp / 100)

                    # Method A: boolean masking (dynamic shape → hidden sync)
                    t_mask = measure(lambda x=x, m=mask: x[m])

                    # Method B: torch.where (fixed output size → no sync)
                    t_where = measure(lambda x=x, m=mask: torch.where(m, x, torch.zeros_like(x)))

                    speedup = t_mask / t_where
                    winner = "where" if speedup > 1 else "mask"
                    log_msg(f"  n={n:>8,}  sparsity={sp:>2}%  mask: {t_mask*1e6:>8.1f}µs  where: {t_where*1e6:>8.1f}µs  {winner} {speedup:.1f}×")

                    csv_w.writerow({"n_elements": n, "sparsity_pct": sp, "method": "boolean_mask", "avg_us": round(t_mask * 1e6, 1), "speedup": round(speedup, 2)})
                    csv_w.writerow({"n_elements": n, "sparsity_pct": sp, "method": "torch.where", "avg_us": round(t_where * 1e6, 1), "speedup": 1.0})

                    del x, mask

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
