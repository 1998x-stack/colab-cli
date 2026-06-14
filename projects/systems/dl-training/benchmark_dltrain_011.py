"""Benchmark dltrain-011: Non-contiguous view vs reshape — silent data corruption.

view() requires contiguous memory; fails silently or crashes on permuted tensors.
reshape() copies when needed. Prefer reshape() unless you explicitly want view-only error.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-011")


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

        log_msg("dltrain-011: view vs reshape on non-contiguous tensors")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["shape", "op", "success", "is_contiguous", "note"])
            csv_w.writeheader()

            shapes = [(2, 3, 4), (4, 4, 4), (8, 16, 32)]

            for shape in shapes:
                x = torch.randn(shape)
                x_t = x.permute(2, 0, 1)  # Now non-contiguous

                log_msg(f"\n{shape} -> permute(2,0,1) -> strides={x_t.stride()}  contiguous={x_t.is_contiguous()}")

                # view on non-contiguous
                try:
                    v = x_t.view(-1)
                    log_msg(f"  view(-1):       OK  shape={list(v.shape)} (silently wrong if non-contiguous!)")
                    csv_w.writerow({"shape": str(shape), "op": "view on non-contiguous", "success": True, "is_contiguous": x_t.is_contiguous(), "note": "silent — may have wrong data layout"})
                except RuntimeError as e:
                    log_msg(f"  view(-1):       ERROR: {str(e)[:80]}")
                    csv_w.writerow({"shape": str(shape), "op": "view on non-contiguous", "success": False, "is_contiguous": x_t.is_contiguous(), "note": str(e)[:80]})

                # reshape on non-contiguous (should always work by copying if needed)
                try:
                    r = x_t.reshape(-1)
                    log_msg(f"  reshape(-1):    OK  shape={list(r.shape)} (copied if needed)")
                    csv_w.writerow({"shape": str(shape), "op": "reshape on non-contiguous", "success": True, "is_contiguous": r.is_contiguous(), "note": "safe — copies when needed"})
                except RuntimeError as e:
                    log_msg(f"  reshape(-1):    ERROR: {str(e)[:80]}")

                # contiguous + view (explicit fix)
                _ = x_t.contiguous().view(-1)
                log_msg("  .contiguous().view(-1): OK (explicit fix, always correct)")

                del x, x_t

        log_msg("\nRule: always use reshape() instead of view() on tensors that may be non-contiguous.")
        log_msg("\nDone.")


if __name__ == "__main__":
    main()
