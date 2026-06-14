"""Benchmark precision-001: FP16 eps=1e-8 rounds to zero — silent NaN generator.

FP16 can't represent values < ~6e-5. Adam's default eps=1e-8 rounds to zero,
producing NaN gradients within 50-200 steps. Fix: eps=1e-4 or higher.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/cuda-dark-corners-output/precision-001")
N_STEPS = 500  # enough to see NaN with bad eps


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

        log_msg("precision-001: FP16 eps=1e-8 rounds to zero")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["step", "eps", "loss", "grad_norm", "status"])
            csv_w.writeheader()

            model_fp16 = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            ).cuda().half()  # FP16 model

            x = torch.randn(64, 256, device="cuda", dtype=torch.float16)
            y = torch.randint(0, 10, (64,), device="cuda")
            loss_fn = torch.nn.CrossEntropyLoss()

            for eps_val, label in [(1e-8, "eps=1e-8 (FP32 default, BROKEN for FP16)"), (1e-4, "eps=1e-4 (FP16-safe)")]:
                # Reset model
                def init_weights(m):
                    if isinstance(m, torch.nn.Linear):
                        m.reset_parameters()
                model_fp16.apply(init_weights)

                opt = torch.optim.Adam(model_fp16.parameters(), lr=1e-3, eps=eps_val)

                log_msg(f"\n--- {label} ---")
                nan_step = None
                for step in range(N_STEPS):
                    opt.zero_grad()
                    out = model_fp16(x)
                    loss = loss_fn(out, y)
                    loss.backward()

                    total_norm = torch.nn.utils.clip_grad_norm_(model_fp16.parameters(), 10.0)
                    grad_norm = float(total_norm) if total_norm is not None else -1

                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        log_msg(f"  Step {step}: NaN/Inf grad norm! Loss={loss.item():.4f}")
                        csv_w.writerow({"step": step, "eps": label, "loss": loss.item(), "grad_norm": -1, "status": "NaN"})
                        nan_step = step
                        break

                    opt.step()

                    if step % 50 == 0 or step == N_STEPS - 1:
                        log_msg(f"  Step {step:>4}: loss={loss.item():.4f}  grad_norm={grad_norm:.6f}")
                        csv_w.writerow({"step": step, "eps": label, "loss": loss.item(), "grad_norm": round(grad_norm, 6), "status": "OK"})

                if nan_step is None:
                    log_msg(f"  Survived {N_STEPS} steps without NaN")

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
