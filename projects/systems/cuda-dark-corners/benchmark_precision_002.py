"""Benchmark precision-002: Tensor Core utilization — AMP vs FP32 matmul size sweep.

T4 Tensor Cores: 65 TFLOPS FP16 theoretical vs 8.1 FP32 (8× cliff).
Real models achieve only 3-4×. Find the matmul sizes where Tensor Cores kick in.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/precision-002")
WARMUP = 10
ITERS = 30

SIZES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


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

        log_msg("precision-002: Tensor Core utilization — AMP vs FP32 matmul sweep")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        log_msg("T4 theoretical: 65 TFLOPS FP16, 8.1 TFLOPS FP32, 130 Tensor core TFLOPS")
        log_msg(f"{'Size':>6s}  {'FP32_ms':>10s}  {'FP32_TFLOPS':>12s}  {'FP16_ms':>10s}  {'FP16_TFLOPS':>12s}  {'Speedup':>8s}  {'Util%':>6s}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["size", "precision", "avg_ms", "tflops", "speedup"])
            csv_w.writeheader()

            for n in SIZES:
                a_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
                b_fp32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
                a_fp16 = a_fp32.half()
                b_fp16 = b_fp32.half()

                t_fp32 = measure(lambda a=a_fp32, b=b_fp32: a @ b)
                t_fp16 = measure(lambda a=a_fp16, b=b_fp16: a @ b)

                flops = 2 * n**3
                tflops_fp32 = flops / t_fp32 / 1e12
                tflops_fp16 = flops / t_fp16 / 1e12
                speedup = t_fp32 / t_fp16
                util = tflops_fp16 / 65 * 100  # % of theoretical FP16 Tensor Core

                log_msg(f"  {n:>5}  {t_fp32*1000:>9.3f}  {tflops_fp32:>11.2f}  {t_fp16*1000:>9.3f}  {tflops_fp16:>11.2f}  {speedup:>7.1f}×  {util:>5.1f}%")

                csv_w.writerow({"size": n, "precision": "FP32", "avg_ms": round(t_fp32 * 1000, 3), "tflops": round(tflops_fp32, 2), "speedup": 1.0})
                csv_w.writerow({"size": n, "precision": "FP16", "avg_ms": round(t_fp16 * 1000, 3), "tflops": round(tflops_fp16, 2), "speedup": round(speedup, 2)})

                del a_fp32, b_fp32, a_fp16, b_fp16

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
