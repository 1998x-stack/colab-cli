"""Benchmark launch-002: CUDA first-call tax — 1.6s hidden initialization.

First CUDA call takes ~1.6s (context creation, PTX compilation, cuBLAS lazy init).
Subsequent calls take ~30µs. torch.cuda.init() does NOT fully initialize CUDA.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/launch-002")
ITERS = 100


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

        log_msg("launch-002: CUDA first-call tax")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["call_type", "call_num", "latency_us"])
            csv_w.writeheader()

            # --- Test 1: torch.cuda.init() — what does it cover? ---
            log_msg("\n--- torch.cuda.init() coverage ---")
            log_msg("(measuring first op latency before and after init)")

            # torch.cuda.init() itself
            t0 = time.perf_counter()
            torch.cuda.init()
            t_init = time.perf_counter() - t0
            log_msg(f"  torch.cuda.init() call: {t_init*1000:.1f} ms")

            # First actual CUDA op after init
            x = torch.tensor([1.0])
            t0 = time.perf_counter()
            y = x.to("cuda")
            torch.cuda.synchronize()
            t_first_to = time.perf_counter() - t0
            log_msg(f"  First .to('cuda') after init: {t_first_to*1000:.1f} ms")

            # Subsequent op
            x2 = torch.tensor([2.0])
            t0 = time.perf_counter()
            y2 = x2.to("cuda")
            torch.cuda.synchronize()
            t_second_to = time.perf_counter() - t0
            log_msg(f"  Second .to('cuda'): {t_second_to*1000000:.1f} µs")
            log_msg(f"  First vs second ratio: {t_first_to / t_second_to:.0f}×")

            del x, y, x2, y2
            torch.cuda.empty_cache()

            # --- Test 2: First small matmul (cuBLAS lazy init) ---
            log_msg("\n--- cuBLAS lazy initialization ---")

            a = torch.randn(128, 128, device="cuda")
            b = torch.randn(128, 128, device="cuda")

            t0 = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize()
            t_first_matmul = time.perf_counter() - t0
            log_msg(f"  First matmul (128×128): {t_first_matmul*1000:.1f} ms")

            t0 = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize()
            t_second_matmul = time.perf_counter() - t0
            log_msg(f"  Second matmul (128×128): {t_second_matmul*1000000:.1f} µs")
            log_msg(f"  First vs second ratio: {t_first_matmul / t_second_matmul:.0f}×")

            csv_w.writerow({"call_type": "first_matmul", "call_num": 1, "latency_us": round(t_first_matmul * 1e6, 1)})
            csv_w.writerow({"call_type": "second_matmul", "call_num": 2, "latency_us": round(t_second_matmul * 1e6, 1)})

            del a, b, c

            # --- Test 3: Full cold-start measurement (what colab exec sees) ---
            log_msg("\n--- Cold-start to hot latency comparison ---")

            # Simulate 100 calls to see the warmup curve
            x_cold = torch.randn(1000, 1000)
            for i in range(10):
                t0 = time.perf_counter()
                y = x_cold.to("cuda")
                z = y @ y.T
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                tag = "FIRST" if i == 0 else f"call #{i}"
                log_msg(f"  {tag:>8}: {elapsed*1000:.2f} ms")
                csv_w.writerow({"call_type": "to_and_matmul", "call_num": i + 1, "latency_us": round(elapsed * 1e6, 1)})
                del y, z
                if i == 0:
                    torch.cuda.empty_cache()

            del x_cold
            torch.cuda.empty_cache()

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
