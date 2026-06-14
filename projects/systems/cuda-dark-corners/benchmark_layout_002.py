"""Benchmark layout-002: cross_entropy vs log_softmax+gather — 29× layout trap for LLM logits.

F.cross_entropy expects channels in dim=1; LLM logits have vocab in dim=-1.
The internal permute creates non-contiguous tensor -> reduction falls back to slow kernel.
log_softmax + gather avoids the permute.
"""

import torch
import torch.nn.functional as F
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-002")
WARMUP = 20
ITERS = 100


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

        log_msg("layout-002: cross_entropy vs log_softmax+gather for LLM logits")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["batch", "seq_len", "vocab", "method", "avg_ms", "speedup"])
            csv_w.writeheader()

            configs = [
                (1, 128, 50257),
                (2, 512, 50257),
                (4, 1024, 50257),
                (1, 128, 32000),   # typical LLM vocab
                (2, 512, 32000),
            ]

            for batch, seq_len, vocab in configs:
                # LLM-typical layout: (batch, seq_len, vocab) — vocab in dim=-1
                logits = torch.randn(batch, seq_len, vocab, device="cuda")
                targets = torch.randint(0, vocab, (batch, seq_len), device="cuda")

                # Method A: F.cross_entropy with layout fix (.contiguous() or transpose)
                # cross_entropy expects (N, C, ...) — permute needed
                t_ce_fixed = measure(lambda: F.cross_entropy(
                    logits.permute(0, 2, 1).contiguous(), targets, reduction="mean"
                ))

                # Method B: F.cross_entropy without .contiguous() — non-contiguous after permute
                t_ce_noncontig = measure(lambda: F.cross_entropy(
                    logits.permute(0, 2, 1), targets, reduction="mean"
                ))

                # Method C: log_softmax + gather (no permute needed)
                t_ls_gather = measure(lambda: -F.log_softmax(logits, dim=-1)
                                       .gather(-1, targets.unsqueeze(-1))
                                       .squeeze(-1).mean())

                # Method D: cross_entropy with vocab in dim=1 layout (native)
                logits_native = logits.permute(0, 2, 1).contiguous()
                t_ce_native = measure(lambda: F.cross_entropy(logits_native, targets, reduction="mean"))

                speedup_best = t_ce_noncontig / min(t_ce_fixed, t_ls_gather, t_ce_native)

                log_msg(f"\n  B={batch} S={seq_len} V={vocab}:")
                log_msg(f"    CE (permute, non-contiguous): {t_ce_noncontig*1000:.4f} ms  ← baseline")
                log_msg(f"    CE (permute + .contiguous()): {t_ce_fixed*1000:.4f} ms  ({t_ce_noncontig/t_ce_fixed:.1f}× faster)")
                log_msg(f"    log_softmax + gather:         {t_ls_gather*1000:.4f} ms  ({t_ce_noncontig/t_ls_gather:.1f}× faster)")
                log_msg(f"    CE (native dim=1 layout):     {t_ce_native*1000:.4f} ms  ({t_ce_noncontig/t_ce_native:.1f}× faster)")

                for method, t in [("CE_noncontiguous", t_ce_noncontig), ("CE_contiguous", t_ce_fixed), ("log_softmax+gather", t_ls_gather), ("CE_native_layout", t_ce_native)]:
                    csv_w.writerow({"batch": batch, "seq_len": seq_len, "vocab": vocab, "method": method, "avg_ms": round(t * 1000, 4), "speedup": round(t_ce_noncontig / t, 2)})

                del logits, targets, logits_native

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
