"""Benchmark transfer-003: Ad-hoc .pin_memory() before .to() is counterproductive.

tensor.pin_memory().to('cuda', non_blocking=True) is 1.5-2× slower than
tensor.to('cuda') because CUDA already creates a pinned staging buffer internally.
The ad-hoc pin_memory() causes double allocation (user-pinned + CUDA-internal pinned).
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/pytorch-transfer-benchmark-output/transfer-003")
WARMUP = 5
ITERS = 20

SIZES_BYTES = [
    (1 * 1024**2, "1 MB"),
    (10 * 1024**2, "10 MB"),
    (50 * 1024**2, "50 MB"),
    (100 * 1024**2, "100 MB"),
    (500 * 1024**2, "500 MB"),
    (1024**3, "1 GB"),
]


def setup():
    Path(OUT_DIR, "logs").mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR, "pngs").mkdir(parents=True, exist_ok=True)


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

        log_msg("transfer-003: Ad-hoc .pin_memory() before .to() anti-pattern")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "method", "avg_ms", "throughput_gbps", "slowdown"])
            csv_w.writeheader()

            for size_bytes, label in SIZES_BYTES:
                nelem = size_bytes // 4  # float32
                vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                if size_bytes * 3 > vram_free * 0.85:
                    log_msg(f"\n{label} — SKIPPED (insufficient VRAM)")
                    continue

                x_cpu = torch.randn(nelem, dtype=torch.float32)

                # Method 1: direct .to('cuda') — CUDA handles pinned staging internally
                t_direct = measure(lambda t=x_cpu: t.to("cuda"))

                # Method 2: ad-hoc .pin_memory().to('cuda', non_blocking=True) — double pinning
                def adhoc_pin(t=x_cpu):
                    y = t.pin_memory()
                    z = y.to("cuda", non_blocking=True)
                    return z

                t_adhoc = measure(adhoc_pin)

                # Method 3: .to('cuda', non_blocking=True) without explicit pin (CUDA handles it)
                t_nb = measure(lambda t=x_cpu: t.to("cuda", non_blocking=True))

                gbps_direct = size_bytes / t_direct / 1e9
                gbps_adhoc = size_bytes / t_adhoc / 1e9
                gbps_nb = size_bytes / t_nb / 1e9
                slowdown = t_adhoc / t_direct

                log_msg(f"\n{label} ({nelem:,} float32 elements):")
                log_msg(f"  .to('cuda') direct:              {t_direct*1000:.2f} ms  ({gbps_direct:.1f} GB/s)")
                log_msg(f"  .to('cuda', nb=True):             {t_nb*1000:.2f} ms  ({gbps_nb:.1f} GB/s)")
                log_msg(f"  .pin_memory().to('cuda', nb=True): {t_adhoc*1000:.2f} ms  ({gbps_adhoc:.1f} GB/s)")
                log_msg(f"  adhoc-pin slowdown vs direct:     {slowdown:.1f}×")

                for method, t, gbps in [
                    ("to(cuda)", t_direct, gbps_direct),
                    ("to(cuda, nb=True)", t_nb, gbps_nb),
                    ("pin_memory().to(cuda, nb=True)", t_adhoc, gbps_adhoc),
                ]:
                    csv_w.writerow({
                        "size": label, "method": method,
                        "avg_ms": round(t * 1000, 3),
                        "throughput_gbps": round(gbps, 2),
                        "slowdown": round(slowdown if method.startswith("pin") else 1.0, 2),
                    })

                del x_cpu
                torch.cuda.empty_cache()

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
