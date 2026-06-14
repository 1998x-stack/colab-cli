"""Benchmark layout-002: cross_entropy vs log_softmax+gather — layout trap for LLM logits.

F.cross_entropy expects channels in dim=1; LLM logits have vocab in dim=-1.
The internal permute creates non-contiguous tensor → reduction falls back to slow kernel.
log_softmax + gather avoids the permute.
"""

import torch
import torch.nn.functional as F
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/layout-002")
WARMUP = 10
ITERS = 100

# LLM-typical shapes: batch × seq_len × vocab
CONFIGS = [
    (1, 128, 50257),
    (2, 256, 50257),
    (2, 512, 50257),
    (4, 512, 50257),
    (8, 512, 50257),
]


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


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("layout-002: cross_entropy vs log_softmax+gather layout trap")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["shape", "method", "avg_ms", "speedup"])
            csv_w.writeheader()

            for batch, seq_len, vocab in CONFIGS:
                # Standard LLM logits shape: (batch, seq_len, vocab) — class dim is -1
                logits = torch.randn(batch, seq_len, vocab, device="cuda")
                targets = torch.randint(0, vocab, (batch, seq_len), device="cuda")

                # Method 1: cross_entropy with permute (internal: permutes to (B, V, S), non-contiguous)
                def ce_permute(logits=logits, targets=targets):
                    return F.cross_entropy(
                        logits.permute(0, 2, 1),  # (B, V, S) — class dim=1
                        targets,
                    )

                # Method 2: cross_entropy with reshape (alternative hack)
                def ce_reshape(logits=logits, targets=targets):
                    return F.cross_entropy(
                        logits.reshape(-1, vocab),  # (B*S, V)
                        targets.reshape(-1),
                    )

                # Method 3: log_softmax + gather (no layout change needed)
                def ls_gather(logits=logits, targets=targets):
                    log_probs = F.log_softmax(logits, dim=-1)
                    return F.nll_loss(
                        log_probs.reshape(-1, vocab),
                        targets.reshape(-1),
                    )

                t_permute = measure(ce_permute)
                t_reshape = measure(ce_reshape)
                t_ls = measure(ls_gather)

                speedup_reshape = t_permute / t_reshape
                speedup_ls = t_permute / t_ls

                shape_str = f"{batch}×{seq_len}×{vocab}"
                log_msg(f"\n{shape_str}:")
                log_msg(f"  cross_entropy(permute):  {t_permute*1000:.3f} ms")
                log_msg(f"  cross_entropy(reshape):  {t_reshape*1000:.3f} ms ({speedup_reshape:.1f}× vs permute)")
                log_msg(f"  log_softmax+gather:      {t_ls*1000:.3f} ms ({speedup_ls:.1f}× vs permute)")

                csv_w.writerow({"shape": shape_str, "method": "ce_permute", "avg_ms": round(t_permute * 1000, 4), "speedup": 1.0})
                csv_w.writerow({"shape": shape_str, "method": "ce_reshape", "avg_ms": round(t_reshape * 1000, 4), "speedup": round(speedup_reshape, 2)})
                csv_w.writerow({"shape": shape_str, "method": "log_softmax+gather", "avg_ms": round(t_ls * 1000, 4), "speedup": round(speedup_ls, 2)})

                del logits, targets

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
