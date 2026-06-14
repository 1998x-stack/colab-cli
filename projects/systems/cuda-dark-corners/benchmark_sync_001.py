"""Benchmark sync-001: .item() hidden sync tax — per-metric vs batched .tolist().

Every .item() call is secretly cudaStreamSynchronize(). Batched .tolist() reduces N syncs
to 1. At 50K steps with 4 metrics, per-metric .item() adds ~27% to training time.
"""

import torch
import time
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/sync-001")
N_STEPS = 1000


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

        log_msg("sync-001: .item() hidden sync tax")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        # Use sync debug mode to surface hidden syncs
        torch.cuda.set_sync_debug_mode("warn")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["method", "n_metrics", "total_ms", "overhead_pct"])
            csv_w.writeheader()

            model = torch.nn.Linear(512, 512).cuda()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            for n_metrics in [1, 4, 10]:
                log_msg(f"\n--- {n_metrics} metrics, {N_STEPS} steps ---")

                x = torch.randn(64, 512, device="cuda")
                target = torch.randn(64, 512, device="cuda")

                # Method 1: per-metric .item() (bad)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for step in range(N_STEPS):
                    opt.zero_grad()
                    out = model(x)
                    loss = torch.nn.functional.mse_loss(out, target)
                    loss.backward()
                    opt.step()
                    _ = [
                        loss.item(),
                        out.mean().item(),
                        out.std().item(),
                        out.abs().max().item(),
                    ][:n_metrics]
                torch.cuda.synchronize()
                t_per_item = time.perf_counter() - t0

                # Method 2: batched .tolist() (good)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for step in range(N_STEPS):
                    opt.zero_grad()
                    out = model(x)
                    loss = torch.nn.functional.mse_loss(out, target)
                    loss.backward()
                    opt.step()
                    loss_val = loss.detach()
                    out_det = out.detach()
                    vals = [loss_val, out_det.mean(), out_det.std(), out_det.abs().max()][:n_metrics]
                    stacked = torch.stack([v.reshape(1) for v in vals])
                    _ = stacked.tolist()
                torch.cuda.synchronize()
                t_tolist = time.perf_counter() - t0

                overhead_pct = (t_per_item - t_tolist) / t_tolist * 100
                speedup = t_per_item / t_tolist

                log_msg(f"  Per-metric .item(): {t_per_item:.2f}s ({t_per_item/N_STEPS*1000:.2f} ms/step)")
                log_msg(f"  Batched .tolist(): {t_tolist:.2f}s ({t_tolist/N_STEPS*1000:.2f} ms/step)")
                log_msg(f"  Overhead: +{overhead_pct:.0f}% ({speedup:.1f}× slower)")

                csv_w.writerow({"method": "per-item", "n_metrics": n_metrics, "total_ms": round(t_per_item * 1000, 1), "overhead_pct": round(overhead_pct, 1)})
                csv_w.writerow({"method": "tolist", "n_metrics": n_metrics, "total_ms": round(t_tolist * 1000, 1), "overhead_pct": 0.0})

            del model, opt, x, target

        torch.cuda.set_sync_debug_mode("default")
        log_msg("\nDone.")


if __name__ == "__main__":
    main()
