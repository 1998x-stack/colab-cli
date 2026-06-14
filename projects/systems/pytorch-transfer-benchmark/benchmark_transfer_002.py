"""Benchmark transfer-002: .to(device, dtype) combined vs two-step ordering trap.

When calling tensor.to(device='cuda', dtype=float32) on uint8 data, PyTorch casts
on CPU first (expanding data 4×), then transfers to GPU. The two-step approach
(transfer uint8 first, cast on GPU) is 3-8× faster for large tensors.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/pytorch-transfer-benchmark-output/transfer-002")
WARMUP = 5
ITERS = 20

SIZES = [
    (256, 256, "256×256"),
    (512, 512, "512×512"),
    (1024, 1024, "1024×1024"),
    (2048, 2048, "2048×2048"),
    (4096, 4096, "4096×4096"),
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

        log_msg("transfer-002: .to(device, dtype) combined vs two-step ordering")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "method", "avg_ms", "speedup"])
            csv_w.writeheader()

            for h, w, label in SIZES:
                nelem = 3 * h * w  # RGB image
                x_cpu = torch.randint(0, 255, (nelem,), dtype=torch.uint8)
                data_mb = x_cpu.numel() * 1 / 1e6

                # Method 1: combined .to(device, dtype) — casts uint8→float32 on CPU first (4× data)
                t_combined = measure(lambda t=x_cpu: t.to(device="cuda", dtype=torch.float32))

                # Method 2: two-step — transfer uint8 first, cast on GPU
                t_twostep = measure(lambda t=x_cpu: t.to("cuda").to(torch.float32))

                speedup = t_combined / t_twostep
                log_msg(f"\n{label} ({data_mb:.1f} MB uint8 → {data_mb*4:.1f} MB float32):")
                log_msg(f"  combined .to(cuda, float32): {t_combined*1000:.2f} ms")
                log_msg(f"  two-step .to(cuda).to(float32): {t_twostep*1000:.2f} ms")
                log_msg(f"  two-step speedup: {speedup:.1f}×")

                csv_w.writerow({
                    "size": label, "method": "combined",
                    "avg_ms": round(t_combined * 1000, 3), "speedup": 1.0,
                })
                csv_w.writerow({
                    "size": label, "method": "two-step",
                    "avg_ms": round(t_twostep * 1000, 3), "speedup": round(speedup, 2),
                })

                del x_cpu
                torch.cuda.empty_cache()

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
