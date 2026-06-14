"""Benchmark precision-002: Tensor Core utilization — AMP vs FP32 matmul size sweep.

T4 Tensor Cores: 65 TFLOPS FP16 theoretical vs 8.1 FP32 (8× cliff).
Real models achieve only 3-4×. Find matmul sizes where Tensor Cores kick in.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/precision-002")
WARMUP = 10
ITERS = 50

MATMUL_SIZES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


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


def tflops(n, t):
    """Compute TFLOPS for matmul of size n×n × n×n."""
    return (2 * n**3) / t / 1e12


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("precision-002: Tensor Core utilization — AMP vs FP32 matmul sweep")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        log_msg("T4 theoretical: 65 TFLOPS (FP16 TC) | 8.1 TFLOPS (FP32)")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "precision", "avg_ms", "tflops", "speedup_vs_fp32"])
            csv_w.writeheader()

            fp32_tflops = {}
            for n in MATMUL_SIZES:
                a_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
                b_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
                a_fp16 = a_fp32.half()
                b_fp16 = b_fp32.half()

                # FP32 baseline
                t_fp32 = measure(lambda a=a_fp32, b=b_fp32: a @ b)
                tf32 = tflops(n, t_fp32)
                fp32_tflops[n] = tf32

                # FP16 (no AMP — direct half precision)
                t_fp16 = measure(lambda a=a_fp16, b=b_fp16: a @ b)
                tf16 = tflops(n, t_fp16)

                # AMP (autocast)
                def amp_mm(a=a_fp32, b=b_fp32):
                    with torch.amp.autocast("cuda"):
                        return a @ b

                t_amp = measure(amp_mm)
                # autocast internally uses FP16 matmul
                tamp_tf = tflops(n, t_amp)

                speedup_fp16 = t_fp32 / t_fp16
                speedup_amp = t_fp32 / t_amp
                utilization = tf16 / 65 * 100  # % of theoretical FP16 TC peak

                log_msg(f"  {n:>5}  FP32: {t_fp32*1000:8.2f}ms ({tf32:5.1f} TF)  FP16: {t_fp16*1000:8.2f}ms ({tf16:5.1f} TF, {speedup_fp16:.1f}×)  AMP: {t_amp*1000:8.2f}ms ({speedup_amp:.1f}×)  util={utilization:.0f}%")

                csv_w.writerow({"size": str(n), "precision": "FP32", "avg_ms": round(t_fp32 * 1000, 3), "tflops": round(tf32, 2), "speedup_vs_fp32": 1.0})
                csv_w.writerow({"size": str(n), "precision": "FP16", "avg_ms": round(t_fp16 * 1000, 3), "tflops": round(tf16, 2), "speedup_vs_fp32": round(speedup_fp16, 2)})
                csv_w.writerow({"size": str(n), "precision": "AMP", "avg_ms": round(t_amp * 1000, 3), "tflops": round(tamp_tf, 2), "speedup_vs_fp32": round(speedup_amp, 2)})

                del a_fp32, b_fp32, a_fp16, b_fp16
                torch.cuda.empty_cache()

            # Summary
            peak_fp32 = max(fp32_tflops.values())
            peak_size_fp32 = max(fp32_tflops, key=fp32_tflops.get)
            log_msg(f"\nPeak FP32: {peak_fp32:.1f} TFLOPS at {peak_size_fp32}×{peak_size_fp32}")

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
