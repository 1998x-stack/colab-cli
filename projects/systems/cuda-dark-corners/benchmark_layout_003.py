"""Benchmark layout-003: index_select vs regular indexing — dedicated function slower for 2D+.

torch.index_select has optimized fast path only for 1D. For 2D+, x[:, idx] is
2-6× faster because it broadcasts indices and uses pointwise ops instead of a
specialized gather kernel.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-003")
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

        log_msg("layout-003: index_select vs regular indexing")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["dims", "shape", "method", "avg_us", "speedup"])
            csv_w.writeheader()

            configs = [
                ("1D", (10000,), 1000),
                ("2D", (1000, 1000), 100),
                ("2D", (5000, 1000), 100),
                ("3D", (100, 1000, 100), 50),
                ("3D", (500, 500, 100), 50),
            ]

            for dims, shape, n_select in configs:
                shape_str = "×".join(str(s) for s in shape)

                for dim in ([0] if dims == "1D" else [0, 1]):  # test dim=0 and dim=1 for 2D+
                    x = torch.randn(*shape, device="cuda")
                    idx = torch.randint(0, shape[dim], (n_select,), device="cuda")

                    # Method A: index_select (dedicated function)
                    t_isel = measure(lambda x=x, d=dim, i=idx: x.index_select(d, i))

                    # Method B: regular indexing (broadcast + pointwise)
                    if dim == 0:
                        t_reg = measure(lambda x=x, i=idx: x[i])
                    else:
                        t_reg = measure(lambda x=x, i=idx: x[:, i])

                    speedup = t_isel / t_reg
                    winner = "regular idx" if speedup > 1 else "index_select"

                    log_msg(f"  {dims} {shape_str:>20s}  dim={dim} select {n_select}:  index_select={t_isel*1e6:.1f}µs  regular={t_reg*1e6:.1f}µs  {winner} {speedup:.1f}×")

                    csv_w.writerow({"dims": dims, "shape": f"{shape_str} dim={dim}", "method": "index_select", "avg_us": round(t_isel * 1e6, 1), "speedup": round(speedup, 2)})
                    csv_w.writerow({"dims": dims, "shape": f"{shape_str} dim={dim}", "method": "regular_indexing", "avg_us": round(t_reg * 1e6, 1), "speedup": 1.0})

                    del x, idx

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
