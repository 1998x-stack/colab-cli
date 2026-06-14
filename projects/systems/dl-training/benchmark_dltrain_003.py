"""Benchmark dltrain-003: OneCycleLR vs constant LR — superconvergence demo.

Train same model with OneCycleLR vs constant LR. OneCycleLR reaches SOTA accuracy
in ~1/10 the iterations by cycling LR up then cosine-annealing down.
Uses CIFAR-10 + small CNN.
"""

import torch
import torch.nn.functional as F
import os
import csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-003")
EPOCHS = 20
BATCH_SIZE = 128
LR_MAX = 0.01


def setup():
    for sub in ["logs", "pngs"]:
        Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)


def get_data():
    import torchvision.transforms as T
    import torchvision.datasets as D
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            print(msg, flush=True)
            log_fh.write(msg + "\n")

        log_msg("dltrain-003: OneCycleLR vs constant LR")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")

        train_loader = get_data()
        steps_per_epoch = len(train_loader)
        total_steps = EPOCHS * steps_per_epoch

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=["scheduler", "epoch", "train_loss", "train_acc", "lr"])
            csv_w.writeheader()

            for sched_name in ["constant", "onecycle"]:
                log_msg(f"\n{'='*40}")
                log_msg(f"Scheduler: {sched_name}")
                log_msg(f"{'='*40}")

                model = SmallCNN().cuda()
                opt = torch.optim.Adam(model.parameters(), lr=LR_MAX if sched_name == "constant" else LR_MAX / 25)
                scaler = torch.amp.GradScaler("cuda")

                if sched_name == "onecycle":
                    sched = torch.optim.lr_scheduler.OneCycleLR(
                        opt, max_lr=LR_MAX, total_steps=total_steps,
                        pct_start=0.3, anneal_strategy="cos",
                    )
                else:
                    sched = None

                for epoch in range(EPOCHS):
                    train_loss, train_acc = train_one_epoch(model, train_loader, opt, scaler, "cuda")
                    if sched:
                        for _ in range(steps_per_epoch):
                            sched.step()
                    lr = opt.param_groups[0]["lr"]
                    log_msg(f"  epoch {epoch+1:>2}: loss={train_loss:.4f}  acc={train_acc:.3f}  lr={lr:.2e}")
                    csv_w.writerow({"scheduler": sched_name, "epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4), "lr": f"{lr:.2e}"})

                del model, opt, scaler
                if sched:
                    del sched

        log_msg("\nDone.")


if __name__ == "__main__":
    main()
