"""Benchmark dltrain-001: Overfit a single batch — the ultimate debug check.

Verify model can memorize 2-16 examples to near-zero loss. Catches: wrong loss
function, bad init, broken data pipeline, gradient flow issues.
"""

import torch
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-001")
N_SAMPLES = 16
N_STEPS = 200


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

        log_msg("dltrain-001: Overfit single batch sanity check")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        # Fixed batch — model should memorize it perfectly
        torch.manual_seed(42)
        x = torch.randn(N_SAMPLES, 128, device="cuda")
        y = torch.randint(0, 10, (N_SAMPLES,), device="cuda")

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ).cuda()

        # No dropout, no weight decay, no augmentation
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        loss_fn = torch.nn.CrossEntropyLoss()

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["step", "loss", "accuracy"])
            csv_w.writeheader()

            initial_loss = None
            for step in range(N_STEPS):
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

                acc = (out.argmax(-1) == y).float().mean().item()

                if step == 0:
                    initial_loss = loss.item()

                if step % 20 == 0 or step == N_STEPS - 1:
                    log_msg(f"  step {step:>4}: loss={loss.item():.6f}  acc={acc:.3f}")
                csv_w.writerow({"step": step, "loss": round(loss.item(), 6), "accuracy": round(acc, 4)})

            log_msg(f"\n  Initial loss: {initial_loss:.4f}")
            log_msg(f"  Final loss:   {loss.item():.6f}")
            log_msg(f"  Final acc:    {acc:.3f}")
            if loss.item() < 0.01:
                log_msg("  VERDICT: PASS — model can overfit single batch")
            elif loss.item() < 0.1:
                log_msg("  VERDICT: WARNING — loss decreasing but not <0.01, check model capacity")
            else:
                log_msg("  VERDICT: FAIL — model cannot overfit, check architecture/loss/optimizer")

            del model, opt, x, y

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
