"""Benchmark launch-001: GPU-vs-CPU crossover point for matmul & element-wise ops.

Finds the exact tensor size where GPU starts beating CPU on T4.
Expected crossover: ~100×100 for matmul, ~10K elements for element-wise ops.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/launch-001")
WARMUP = 10
ITERS = 50

MATMUL_SIZES = [16, 32, 64, 100, 128, 200, 256, 512, 1024, 2048]
ELEM_SIZES = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def measure(fn, device, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
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

        log_msg("launch-001: GPU-vs-CPU crossover point")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        log_msg(f"CPU: {os.cpu_count()} cores")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["op", "size", "device", "avg_ms", "speedup_vs_cpu"])
            csv_w.writeheader()

            # --- Matmul benchmark ---
            log_msg("\n" + "=" * 60)
            log_msg("MATMUL: CPU vs GPU crossover")
            log_msg("=" * 60)

            for n in MATMUL_SIZES:
                a_cpu = torch.randn(n, n)
                b_cpu = torch.randn(n, n)

                t_cpu = measure(lambda a=a_cpu, b=b_cpu: a @ b, "cpu")
                flops = 2 * n**3
                gflops_cpu = flops / t_cpu / 1e9

                a_gpu = a_cpu.to("cuda")
                b_gpu = b_cpu.to("cuda")
                t_gpu = measure(lambda a=a_gpu, b=b_gpu: a @ b, "cuda")
                gflops_gpu = flops / t_gpu / 1e9

                speedup = t_cpu / t_gpu
                winner = "GPU" if speedup > 1 else "CPU"
                crossover_mark = " <<< CROSSOVER" if (n >= 64 and speedup > 1 and speedup < 10) else ""
                log_msg(f"  {n:>5}×{n:<5}  CPU: {t_cpu*1000:8.2f} ms ({gflops_cpu:.1f} GFLOPS)  |  GPU: {t_gpu*1000:8.2f} ms ({gflops_gpu:.1f} GFLOPS)  |  {winner} {speedup:.1f}×{crossover_mark}")

                csv_w.writerow({"op": "matmul", "size": f"{n}×{n}", "device": "cpu", "avg_ms": round(t_cpu * 1000, 4), "speedup_vs_cpu": 1.0})
                csv_w.writerow({"op": "matmul", "size": f"{n}×{n}", "device": "cuda", "avg_ms": round(t_gpu * 1000, 4), "speedup_vs_cpu": round(speedup, 2)})

                del a_cpu, b_cpu, a_gpu, b_gpu
                torch.cuda.empty_cache()

            # --- Element-wise benchmark ---
            log_msg("\n" + "=" * 60)
            log_msg("ELEMENT-WISE (relu): CPU vs GPU crossover")
            log_msg("=" * 60)

            for n in ELEM_SIZES:
                x_cpu = torch.randn(n)
                t_cpu = measure(lambda x=x_cpu: torch.relu(x), "cpu")

                x_gpu = x_cpu.to("cuda")
                t_gpu = measure(lambda x=x_gpu: torch.relu(x), "cuda")

                speedup = t_cpu / t_gpu
                winner = "GPU" if speedup > 1 else "CPU"
                crossover_mark = " <<< CROSSOVER" if (n >= 5000 and speedup > 1 and speedup < 5) else ""
                log_msg(f"  {n:>9,} elems  CPU: {t_cpu*1000:8.3f} ms  |  GPU: {t_gpu*1000:8.3f} ms  |  {winner} {speedup:.1f}×{crossover_mark}")

                csv_w.writerow({"op": "relu", "size": str(n), "device": "cpu", "avg_ms": round(t_cpu * 1000, 4), "speedup_vs_cpu": 1.0})
                csv_w.writerow({"op": "relu", "size": str(n), "device": "cuda", "avg_ms": round(t_gpu * 1000, 4), "speedup_vs_cpu": round(speedup, 2)})

                del x_cpu, x_gpu
                torch.cuda.empty_cache()

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
