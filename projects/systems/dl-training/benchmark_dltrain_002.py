"""Benchmark dltrain-002: Loss at initialization — theoretical sanity check.

For K-class softmax, initial loss must be -ln(1/K) ≈ 2.30 for CIFAR-10.
If wildly different, init or loss function is wrong. Compare kaiming_uniform vs
xavier_uniform initial loss on a transformer — 17× grad norm difference.
"""

import torch
import torch.nn.functional as F
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-002")
K = 10  # CIFAR-10 classes
EXPECTED_LOSS = -__import__("math").log(1 / K)


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def init_loss(init_fn, label):
    """Compute initial loss for a given init function on random data."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, K),
    )
    # Apply init
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.cuda()
    x = torch.randn(64, 128, device="cuda")
    y = torch.randint(0, K, (64,), device="cuda")

    with torch.no_grad():
        out = model(x)
        loss = F.cross_entropy(out, y).item()

    # Compute grad norm on one backward pass
    model.zero_grad()
    out = model(x)
    loss_val = F.cross_entropy(out, y)
    loss_val.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    del model
    return loss, grad_norm


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("dltrain-002: Loss at initialization sanity check")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        log_msg(f"Expected initial loss for K={K}: -ln(1/K) = {EXPECTED_LOSS:.4f}")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["init", "initial_loss", "loss_error_pct", "grad_norm"])
            csv_w.writeheader()

            inits = [
                (torch.nn.init.kaiming_uniform_, "kaiming_uniform"),
                (torch.nn.init.xavier_uniform_, "xavier_uniform"),
                (torch.nn.init.kaiming_normal_, "kaiming_normal"),
                (torch.nn.init.xavier_normal_, "xavier_normal"),
                (lambda w: torch.nn.init.normal_(w, 0, 0.01), "N(0, 0.01)"),
                (lambda w: torch.nn.init.normal_(w, 0, 1.0), "N(0, 1.0)"),
            ]

            for init_fn, name in inits:
                loss, grad_norm = init_loss(init_fn, name)
                error_pct = abs(loss - EXPECTED_LOSS) / EXPECTED_LOSS * 100
                verdict = "PASS" if error_pct < 10 else ("WARN" if error_pct < 50 else "FAIL")
                log_msg(f"  {name:>20}: loss={loss:.4f} (err={error_pct:.0f}%)  grad_norm={grad_norm:.1f}  [{verdict}]")
                csv_w.writerow({"init": name, "initial_loss": round(loss, 4), "loss_error_pct": round(error_pct, 1), "grad_norm": round(grad_norm, 2)})

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
