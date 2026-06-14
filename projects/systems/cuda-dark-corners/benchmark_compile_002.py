"""Benchmark compile-002: Non-contiguous max() under torch.compile — 8× layout sensitivity.

torch.max(x) when x is non-contiguous triggers 3-stage reduction instead of 2-stage
under inductor. Same values, 8× slower. A .contiguous() call before .max() fixes it.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/compile-002")
WARMUP = 20
ITERS = 100

SIZES = [256, 512, 1024, 2048, 4096]


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

        log_msg("compile-002: Non-contiguous max() under torch.compile")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "mode", "layout", "avg_us", "slowdown"])
            csv_w.writeheader()

            for n in SIZES:
                x_contig = torch.randn(n, n, device="cuda")
                x_trans = x_contig.T.contiguous().T  # non-contiguous

                # --- Eager mode ---
                # contiguous max
                t_eager_contig = measure(lambda x=x_contig: torch.max(x))
                # non-contiguous max
                t_eager_trans = measure(lambda x=x_trans: torch.max(x))
                # fix: .contiguous() before max
                t_eager_fixed = measure(lambda x=x_trans: torch.max(x.contiguous()))

                # --- Compile mode ---
                # Compile each variant
                max_compiled_contig = torch.compile(lambda x: torch.max(x))
                max_compiled_trans = torch.compile(lambda x: torch.max(x))
                max_compiled_fixed = torch.compile(lambda x: torch.max(x.contiguous()))

                # Warmup compiled
                for _ in range(5):
                    max_compiled_contig(x_contig)
                    max_compiled_trans(x_trans)
                    max_compiled_fixed(x_trans)
                torch.cuda.synchronize()

                t_comp_contig = measure(lambda x=x_contig: max_compiled_contig(x))
                t_comp_trans = measure(lambda x=x_trans: max_compiled_trans(x))
                t_comp_fixed = measure(lambda x=x_trans: max_compiled_fixed(x))

                slowdown_eager = t_eager_trans / t_eager_contig
                slowdown_comp = t_comp_trans / t_comp_contig

                log_msg(f"\n{n}×{n}:")
                log_msg(f"  Eager  contiguous:     {t_eager_contig*1e6:8.1f}µs")
                log_msg(f"  Eager  non-contiguous: {t_eager_trans*1e6:8.1f}µs ({slowdown_eager:.1f}×)")
                log_msg(f"  Eager  fixed:          {t_eager_fixed*1e6:8.1f}µs")
                log_msg(f"  Comp   contiguous:     {t_comp_contig*1e6:8.1f}µs")
                log_msg(f"  Comp   non-contiguous: {t_comp_trans*1e6:8.1f}µs ({slowdown_comp:.1f}×)")
                log_msg(f"  Comp   fixed:          {t_comp_fixed*1e6:8.1f}µs")

                csv_w.writerow({"size": f"{n}×{n}", "mode": "eager", "layout": "contiguous", "avg_us": round(t_eager_contig * 1e6, 1), "slowdown": 1.0})
                csv_w.writerow({"size": f"{n}×{n}", "mode": "eager", "layout": "non-contiguous", "avg_us": round(t_eager_trans * 1e6, 1), "slowdown": round(slowdown_eager, 2)})
                csv_w.writerow({"size": f"{n}×{n}", "mode": "compile", "layout": "contiguous", "avg_us": round(t_comp_contig * 1e6, 1), "slowdown": 1.0})
                csv_w.writerow({"size": f"{n}×{n}", "mode": "compile", "layout": "non-contiguous", "avg_us": round(t_comp_trans * 1e6, 1), "slowdown": round(slowdown_comp, 2)})

                del x_contig, x_trans

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
