"""Benchmark layout-001: Implicit .contiguous() copies from chained ops on transposed tensors.

After .T/.permute(), many ops silently call .contiguous() triggering full data copies.
A chain of 10 ops on a non-contiguous tensor can copy 5-15 times without user knowing.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-001")
WARMUP = 10
ITERS = 50


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


def count_contiguous_calls(fn):
    """Count aten::copy_ calls triggered by implicit .contiguous()."""
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        fn()
    count = 0
    for event in prof.key_averages():
        if "copy_" in event.key:
            count += event.count
    return count


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("layout-001: Implicit .contiguous() copies")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "layout", "op_chain", "avg_ms", "slowdown"])
            csv_w.writeheader()

            sizes = [256, 512, 1024, 2048]
            n_layers = 5  # chain depth

            for n in sizes:
                x = torch.randn(n, n, device="cuda")

                # Contiguous version
                def chain_contiguous(t=x):
                    for _ in range(n_layers):
                        t = t @ t.T + t.relu()
                    return t

                # Non-contiguous version: transpose first, then same chain
                x_t = x.T.contiguous().T  # force non-contiguous
                def chain_noncontiguous(t=x_t):
                    for _ in range(n_layers):
                        t = t @ t.T + t.relu()
                    return t

                t_cont = measure(chain_contiguous)
                t_noncont = measure(chain_noncontiguous)
                slowdown = t_noncont / t_cont

                log_msg(f"\n  {n}×{n}, {n_layers}-layer chain:")
                log_msg(f"    contiguous:     {t_cont*1000:.3f} ms")
                log_msg(f"    non-contiguous: {t_noncont*1000:.3f} ms")
                log_msg(f"    slowdown:       {slowdown:.1f}×")

                csv_w.writerow({"size": f"{n}×{n}", "layout": "contiguous", "op_chain": f"{n_layers}×op", "avg_ms": round(t_cont * 1000, 3), "slowdown": 1.0})
                csv_w.writerow({"size": f"{n}×{n}", "layout": "non-contiguous", "op_chain": f"{n_layers}×op", "avg_ms": round(t_noncont * 1000, 3), "slowdown": round(slowdown, 1)})

                # Profile one case to count implicit copies
                if n == 512:
                    n_copies = count_contiguous_calls(lambda: chain_noncontiguous(x_t.clone()))
                    log_msg(f"    implicit copy_ calls in {n_layers}× chain: {n_copies}")

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
