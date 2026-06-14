"""Benchmark layout-001: Implicit .contiguous() copies on transposed tensors.

After .T/.permute(), many ops silently call .contiguous() triggering full data copies.
A chain of 10 ops on a non-contiguous tensor can copy 5-15 times without user knowing.
Uses torch.autograd.profiler to count aten::copy_ calls.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-001")
WARMUP = 5
ITERS = 30

SIZES = [256, 512, 1024, 2048, 4096]


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def count_copies(fn):
    """Count aten::copy_ calls during fn execution."""
    with torch.autograd.profiler.profile() as prof:
        fn()
    return sum(1 for e in prof.key_averages() if "copy_" in e.key)


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


def chain_ops(x):
    """A realistic op chain: matmul, layer_norm, add, relu, matmul repeated."""
    w1 = torch.randn(x.size(-1), x.size(-1), device=x.device)
    w2 = torch.randn(x.size(-1), x.size(-1), device=x.device)
    y = x @ w1
    y = torch.layer_norm(y, [y.size(-1)])
    y = y + 0.1
    y = torch.relu(y)
    y = y @ w2
    y = torch.layer_norm(y, [y.size(-1)])
    y = y + 0.1
    y = torch.relu(y)
    y = y @ w1
    return y


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
            csv_w = csv.DictWriter(cf, fieldnames=["size", "layout", "avg_ms", "copy_count", "slowdown"])
            csv_w.writeheader()

            for n in SIZES:
                x_contig = torch.randn(n, n, device="cuda")
                x_trans = x_contig.T.contiguous().T  # force non-contiguous but keep values

                log_msg(f"\n--- {n}×{n} ---")
                log_msg(f"  contiguous:     strides={x_contig.stride()}")
                log_msg(f"  non-contiguous: strides={x_trans.stride()}")

                # Contiguous baseline
                t_contig = measure(lambda t=x_contig: chain_ops(t))
                copies_contig = count_copies(lambda t=x_contig: chain_ops(t))

                # Non-contiguous
                t_trans = measure(lambda t=x_trans: chain_ops(t))
                copies_trans = count_copies(lambda t=x_trans: chain_ops(t))

                slowdown = t_trans / t_contig
                log_msg(f"  contiguous:     {t_contig*1000:.2f} ms  |  copy_ calls: {copies_contig}")
                log_msg(f"  non-contiguous: {t_trans*1000:.2f} ms  |  copy_ calls: {copies_trans}")
                log_msg(f"  slowdown: {slowdown:.1f}×  |  extra copies: {copies_trans - copies_contig}")

                csv_w.writerow({"size": f"{n}×{n}", "layout": "contiguous", "avg_ms": round(t_contig * 1000, 2), "copy_count": copies_contig, "slowdown": 1.0})
                csv_w.writerow({"size": f"{n}×{n}", "layout": "non-contiguous", "avg_ms": round(t_trans * 1000, 2), "copy_count": copies_trans, "slowdown": round(slowdown, 2)})

                del x_contig, x_trans

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
