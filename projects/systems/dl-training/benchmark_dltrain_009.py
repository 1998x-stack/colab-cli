"""dltrain-009: LR Finder — find optimal LR in one epoch.

Start with tiny LR, exponentially increase each batch, plot loss vs LR.
Pick LR at steepest descent (~1/10 of divergence point).
"""
import torch, torch.nn.functional as F, os, csv, math
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-009")
LR_MIN, LR_MAX = 1e-7, 10.0

def setup():
    for sub in ["logs", "pngs"]: Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)

def get_data():
    import torchvision.transforms as T, torchvision.datasets as D
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, 10),
        )
    def forward(self, x): return self.net(x)

def main():
    setup(); loader = get_data()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["batch", "lr", "loss", "smoothed_loss"])
        csv_w.writeheader()
        log_msg("dltrain-009: LR Finder")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        model = CNN().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=LR_MIN)
        n_batches = len(loader)
        mult = (LR_MAX / LR_MIN) ** (1.0 / n_batches)
        log_msg(f"Exploring LR from {LR_MIN:.0e} to {LR_MAX:.0e} over {n_batches} batches")
        losses, lrs = [], []
        best_loss, best_lr = float("inf"), LR_MIN
        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda(); opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if torch.isnan(loss) or loss.item() > best_loss * 4:
                log_msg(f"  Diverged at batch {batch_idx}, LR={opt.param_groups[0]['lr']:.2e}")
                break
            loss.backward(); opt.step()
            lr = opt.param_groups[0]["lr"]
            losses.append(loss.item()); lrs.append(lr)
            if loss.item() < best_loss: best_loss, best_lr = loss.item(), lr
            opt.param_groups[0]["lr"] *= mult
            if batch_idx % 50 == 0:
                log_msg(f"  batch {batch_idx:>4}: lr={lr:.2e}  loss={loss.item():.4f}")
        # Smooth and find steepest descent
        smoothed = [losses[0]]
        for v in losses[1:]: smoothed.append(0.9 * smoothed[-1] + 0.1 * v)
        min_smoothed_idx = min(range(len(smoothed)), key=lambda i: smoothed[i])
        recommended_lr = lrs[min_smoothed_idx] / 10
        for i in range(0, len(lrs), max(1, len(lrs)//20)):
            csv_w.writerow({"batch": i, "lr": f"{lrs[i]:.2e}", "loss": round(losses[i], 4), "smoothed_loss": round(smoothed[i], 4)})
        log_msg(f"\nBest loss: {best_loss:.4f} at LR={best_lr:.2e}")
        log_msg(f"Min smoothed loss at LR={lrs[min_smoothed_idx]:.2e}")
        log_msg(f"Recommended LR: {recommended_lr:.2e} (1/10 of min point)")
        log_msg("\nDone.")

if __name__ == "__main__": main()
