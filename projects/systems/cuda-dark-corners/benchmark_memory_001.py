"""Benchmark memory-001: Allocator fragmentation — allocation order causes OOM.

Allocating small-then-large can waste 2× reserved memory because small segments
can't merge. expandable_segments:True eliminates this. Same bytes, different order
→ OOM or not.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/memory-001")


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def get_mem_info():
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("memory-001: Allocator fragmentation — allocation order causes OOM")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        log_msg(f"Total VRAM: {total_mb:.0f} MB")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["test", "order", "allocated_mb", "reserved_mb", "waste_pct", "result"])
            csv_w.writeheader()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # --- Test 1: Small-then-large (BAD order — causes fragmentation) ---
            log_msg("\n--- Test 1: Small-then-large (fragmentation-prone) ---")
            torch.cuda.empty_cache()
            small_tensors = []
            for _ in range(20):
                small_tensors.append(torch.zeros(256, 256, device="cuda"))  # 256KB each
            mem_after_small = get_mem_info()
            log_msg(f"  After 20× small tensors: allocated={mem_after_small['allocated_mb']:.0f}MB  reserved={mem_after_small['reserved_mb']:.0f}MB")

            # Try to allocate large tensor
            block_mb = int(total_mb * 0.45)  # 45% of VRAM
            nelem = block_mb * 1024 * 1024 // 4
            try:
                large = torch.zeros(nelem, device="cuda")
                mem_after_large = get_mem_info()
                waste = (mem_after_large['reserved_mb'] - mem_after_large['allocated_mb']) / mem_after_large['allocated_mb'] * 100
                log_msg(f"  After large ({block_mb}MB): allocated={mem_after_large['allocated_mb']:.0f}MB  reserved={mem_after_large['reserved_mb']:.0f}MB  waste={waste:.0f}%")
                csv_w.writerow({"test": "small-then-large", "order": "bad", "allocated_mb": round(mem_after_large['allocated_mb'], 1), "reserved_mb": round(mem_after_large['reserved_mb'], 1), "waste_pct": round(waste, 1), "result": "OK"})
                del large
            except RuntimeError as e:
                log_msg(f"  OOM on large tensor: {e}")
                csv_w.writerow({"test": "small-then-large", "order": "bad", "allocated_mb": round(mem_after_small['allocated_mb'], 1), "reserved_mb": round(mem_after_small['reserved_mb'], 1), "waste_pct": 0, "result": "OOM"})

            del small_tensors
            torch.cuda.empty_cache()

            # --- Test 2: Large-then-small (GOOD order — no fragmentation) ---
            log_msg("\n--- Test 2: Large-then-small (fragmentation-resistant) ---")
            torch.cuda.reset_peak_memory_stats()

            try:
                large_first = torch.zeros(nelem, device="cuda")
                mem_after_large = get_mem_info()
                log_msg(f"  After large ({block_mb}MB): allocated={mem_after_large['allocated_mb']:.0f}MB  reserved={mem_after_large['reserved_mb']:.0f}MB")

                small2 = []
                for _ in range(20):
                    small2.append(torch.zeros(256, 256, device="cuda"))
                mem_final = get_mem_info()
                waste2 = (mem_final['reserved_mb'] - mem_final['allocated_mb']) / mem_final['allocated_mb'] * 100
                log_msg(f"  After 20× small: allocated={mem_final['allocated_mb']:.0f}MB  reserved={mem_final['reserved_mb']:.0f}MB  waste={waste2:.0f}%")
                csv_w.writerow({"test": "large-then-small", "order": "good", "allocated_mb": round(mem_final['allocated_mb'], 1), "reserved_mb": round(mem_final['reserved_mb'], 1), "waste_pct": round(waste2, 1), "result": "OK"})
                del large_first, small2
            except RuntimeError as e:
                log_msg(f"  OOM: {e}")
                csv_w.writerow({"test": "large-then-small", "order": "good", "allocated_mb": 0, "reserved_mb": 0, "waste_pct": 0, "result": "OOM"})

            torch.cuda.empty_cache()

            # --- Summary ---
            log_msg(f"\n--- Summary ---")
            log_msg(f"  Peak allocated:  {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
            log_msg(f"  Peak reserved:   {torch.cuda.max_memory_reserved() / 1024**2:.0f} MB")
            log_msg(f"  Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate fragmentation.")

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
