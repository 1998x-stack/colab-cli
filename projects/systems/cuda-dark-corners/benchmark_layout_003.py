"""Benchmark layout-003: index_select vs regular indexing — dedicated function slower for 2D+.

torch.index_select has optimized fast path only for 1D. For 2D+, x[:, idx] is 2-6×
faster because it broadcasts indices and uses pointwise ops instead of gather kernel.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-003")
WARMUP = 10
ITERS = 100

# (dim0, dim1, n_select) — varying tensor shapes and select counts
CONFIGS = [
    (100, 128, 10, "1D"),     # 1D case — index_select should be fine
    (100, 128, 10, "2D-rows"),  # select rows from 2D
    (500, 256, 50, "2D-rows"),
    (100, 128, 10, "2D-cols"),  # select cols from 2D
    (500, 256, 50, "2D-cols"),
    (128, 64, 32, "3D-batch"), # select from first dim of 3D
]


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def measure(fn):
    for _ in range(WARMUP):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


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
            csv_w = csv.DictWriter(cf, fieldnames=["shape", "dim", "method", "avg_us", "speedup"])
            csv_w.writeheader()

            for d0, d1, n_sel, label in CONFIGS:
                idx = torch.randint(0, d1 if "cols" in label else d0, (n_sel,), device="cuda")

                if label == "1D":
                    x = torch.randn(d0, device="cuda")
                    dim = 0
                    t_is = measure(lambda xx=x, ii=idx: torch.index_select(xx, dim, ii))
                    t_reg = measure(lambda xx=x, ii=idx: xx[ii])
                elif "rows" in label:
                    x = torch.randn(d0, d1, device="cuda")
                    dim = 0
                    t_is = measure(lambda xx=x, ii=idx: torch.index_select(xx, dim, ii))
                    t_reg = measure(lambda xx=x, ii=idx: xx[ii])
                elif "cols" in label:
                    x = torch.randn(d0, d1, device="cuda")
                    dim = 1
                    idx = torch.randint(0, d1, (n_sel,), device="cuda")
                    t_is = measure(lambda xx=x, ii=idx: torch.index_select(xx, dim, ii))
                    t_reg = measure(lambda xx=x, ii=idx: xx[:, ii])
                elif "3D" in label:
                    x = torch.randn(d0, d1, n_sel, device="cuda")
                    dim = 0
                    idx = torch.randint(0, d0, (n_sel,), device="cuda")
                    t_is = measure(lambda xx=x, ii=idx: torch.index_select(xx, dim, ii))
                    t_reg = measure(lambda xx=x, ii=idx: xx[ii])

                speedup = t_is / t_reg

                shape_str = f"{d0}×{d1}" if "3D" not in label else f"{d0}×{d1}×{n_sel}"
                log_msg(f"  {shape_str:>12} dim={dim} select={n_sel}  index_select: {t_is*1e6:8.1f}µs  |  regular: {t_reg*1e6:8.1f}µs  |  {speedup:.1f}× {'(index_select faster)' if speedup < 1 else '(regular faster)'}")

                csv_w.writerow({"shape": shape_str, "dim": dim, "method": "index_select", "avg_us": round(t_is * 1e6, 1), "speedup": round(speedup, 2)})
                csv_w.writerow({"shape": shape_str, "dim": dim, "method": "regular_index", "avg_us": round(t_reg * 1e6, 1), "speedup": 1.0})

                del x, idx

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
