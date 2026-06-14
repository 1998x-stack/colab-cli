"""Benchmark precision-001: FP16 eps=1e-8 rounds to zero — silent NaN generator.

FP16 can't represent values < ~6e-5. Adam's default eps=1e-8 rounds to zero,
producing NaN gradients within 50-200 steps. Fix: eps=1e-4 or higher.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/precision-001")
N_STEPS = 500


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

        log_msg("precision-001: FP16 eps=1e-8 silent NaN generator")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["eps", "step", "loss", "grad_norm", "has_nan"])
            csv_w.writeheader()

            for eps_val in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                log_msg(f"\n{'='*40}")
                log_msg(f"eps = {eps_val:.0e}")
                log_msg(f"{'='*40}")

                model = torch.nn.Sequential(
                    torch.nn.Linear(128, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 10),
                ).cuda()

                opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=eps_val, weight_decay=0)
                scaler = torch.amp.GradScaler("cuda")
                loss_fn = torch.nn.CrossEntropyLoss()

                nan_step = None
                for step in range(N_STEPS):
                    x = torch.randn(32, 128, device="cuda")
                    y = torch.randint(0, 10, (32,), device="cuda")

                    opt.zero_grad()
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                        loss = loss_fn(out, y)

                    scaler.scale(loss).backward()

                    # Check for NaN
                    has_nan = False
                    for p in model.parameters():
                        if p.grad is not None and (not p.grad.isfinite().all()):
                            has_nan = True
                            break

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

                    scaler.step(opt)
                    scaler.update()

                    if step % 50 == 0 or has_nan:
                        log_msg(f"  step {step:>4}: loss={loss.item():.4f}  grad_norm={grad_norm:.4f}  nan={'!!!' if has_nan else 'ok'}")

                    csv_w.writerow({"eps": f"{eps_val:.0e}", "step": step, "loss": round(loss.item(), 6), "grad_norm": round(grad_norm, 6), "has_nan": has_nan})

                    if has_nan:
                        nan_step = step
                        log_msg(f"  -> NaN detected at step {step} with eps={eps_val:.0e}")
                        break

                if nan_step is None:
                    log_msg(f"  -> No NaN within {N_STEPS} steps with eps={eps_val:.0e}")

                del model, opt, scaler

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
