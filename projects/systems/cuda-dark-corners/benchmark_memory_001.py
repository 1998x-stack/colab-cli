"""Benchmark memory-001: Allocator fragmentation — allocation order causes OOM.

Allocating small-then-large can waste 2× reserved memory because small segments
can't merge. expandable_segments:True eliminates this. Same total bytes, different
allocation order -> OOM or not.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/memory-001")


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def alloc_test(label, allocs_fn):
    """Run an allocation pattern and report memory stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        tensors = allocs_fn()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        waste = (reserved - allocated) / allocated * 100 if allocated > 0 else 0
        # Cleanup
        for t in tensors:
            del t
        torch.cuda.empty_cache()
        return True, allocated, reserved, peak, waste
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, 0, 0, 0, 0
        raise


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log_msg("memory-001: Allocator fragmentation")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {total:.1f} GB  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["pattern", "success", "allocated_mb", "reserved_mb", "peak_mb", "waste_pct", "expandable"])
            csv_w.writeheader()

            MB = 1024 * 1024
            n_small = 100
            small_size = 5 * MB  # 5 MB each = 500 MB total
            large_size = 6000 * MB  # 6 GB single tensor
            # Total = 6.5 GB (fits in T4 15.6 GB)
            # But small-then-large fragments: 100 × 5MB interleaved → can't merge

            # --- Pattern 1: small-then-large (fragmentation-prone) ---
            def small_then_large():
                tensors = []
                for _ in range(n_small):
                    tensors.append(torch.randn(small_size // 4, device="cuda"))
                tensors.append(torch.randn(large_size // 4, device="cuda"))
                return tensors

            ok, alloc, res, peak, waste = alloc_test("small-then-large", small_then_large)
            log_msg("\nsmall-then-large (100×5MB + 1×6GB):")
            log_msg(f"  success={ok}  allocated={alloc:.0f}MB  reserved={res:.0f}MB  peak={peak:.0f}MB  waste={waste:.0f}%")
            csv_w.writerow({"pattern": "small-then-large", "success": ok, "allocated_mb": round(alloc, 1) if ok else 0, "reserved_mb": round(res, 1) if ok else 0, "peak_mb": round(peak, 1) if ok else 0, "waste_pct": round(waste, 1) if ok else 0, "expandable": False})

            # --- Pattern 2: large-then-small (fragmentation-avoiding) ---
            def large_then_small():
                tensors = [torch.randn(large_size // 4, device="cuda")]
                for _ in range(n_small):
                    tensors.append(torch.randn(small_size // 4, device="cuda"))
                return tensors

            ok, alloc, res, peak, waste = alloc_test("large-then-small", large_then_small)
            log_msg("\nlarge-then-small (1×6GB + 100×5MB):")
            log_msg(f"  success={ok}  allocated={alloc:.0f}MB  reserved={res:.0f}MB  peak={peak:.0f}MB  waste={waste:.0f}%")
            csv_w.writerow({"pattern": "large-then-small", "success": ok, "allocated_mb": round(alloc, 1) if ok else 0, "reserved_mb": round(res, 1) if ok else 0, "peak_mb": round(peak, 1) if ok else 0, "waste_pct": round(waste, 1) if ok else 0, "expandable": False})

            # --- Pattern 3: interleaved small-large-small (worst case) ---
            def interleaved():
                tensors = []
                for i in range(50):
                    tensors.append(torch.randn(small_size // 4, device="cuda"))
                    tensors.append(torch.randn(large_size // 50 // 4, device="cuda"))
                return tensors

            ok, alloc, res, peak, waste = alloc_test("interleaved", interleaved)
            log_msg("\ninterleaved (50×[5MB + 120MB]):")
            log_msg(f"  success={ok}  allocated={alloc:.0f}MB  reserved={res:.0f}MB  peak={peak:.0f}MB  waste={waste:.0f}%")
            csv_w.writerow({"pattern": "interleaved", "success": ok, "allocated_mb": round(alloc, 1) if ok else 0, "reserved_mb": round(res, 1) if ok else 0, "peak_mb": round(peak, 1) if ok else 0, "waste_pct": round(waste, 1) if ok else 0, "expandable": False})

            # --- Pattern 4: Trigger OOM via fragmentation ---
            # Fill most of VRAM with many small tensors, then try a large one
            log_msg("\n--- Fragmentation stress test ---")
            n_fill = 800
            fill_size = 15 * MB  # 800 × 15MB = 12GB allocated → ~14GB reserved
            big = 3000 * MB

            def fill_then_big():
                tensors = []
                for _ in range(n_fill):
                    tensors.append(torch.randn(fill_size // 4, device="cuda"))
                allocated_mid = torch.cuda.memory_allocated() / 1024**2
                reserved_mid = torch.cuda.memory_reserved() / 1024**2
                log_msg(f"  After {n_fill}×15MB: allocated={allocated_mid:.0f}MB  reserved={reserved_mid:.0f}MB")
                try:
                    tensors.append(torch.randn(big // 4, device="cuda"))
                    log_msg("  Big alloc (3GB): SUCCESS")
                except RuntimeError:
                    log_msg("  Big alloc (3GB): OOM! Fragmentation prevented allocation")
                return tensors

            ok, alloc, res, peak, waste = alloc_test("fill-then-big", fill_then_big)
            csv_w.writerow({"pattern": "fill-then-big", "success": ok, "allocated_mb": round(alloc, 1) if ok else 0, "reserved_mb": round(res, 1) if ok else 0, "peak_mb": round(peak, 1) if ok else 0, "waste_pct": round(waste, 1) if ok else 0, "expandable": False})

            log_msg(f"\nSummary: fragmentation waste = {waste:.0f}% of allocated memory")
            log_msg("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate fragmentation.")

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
