"""dltrain-007: Data augmentation hurts at low epochs — Colab-critical.

With <90 epochs, augmentation slows convergence. At 20 epochs, no-aug beats aug
by 2-5%. Colab GPUs die in ~10 min — augmentation may waste your limited window.
"""
import torch, torch.nn.functional as F, os, csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-007")
EPOCHS = 15; BATCH = 128

def setup():
    for sub in ["logs", "pngs"]: Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)

def get_loaders(with_aug):
    import torchvision.transforms as T, torchvision.datasets as D
    base = [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if with_aug else []
    tf = T.Compose(aug + base)
    ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    test_ds = D.CIFAR10(root="/content/data", train=False, download=True, transform=T.Compose(base))
    return (torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True),
            torch.utils.data.DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True))

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
    setup()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["augmentation", "epoch", "train_loss", "train_acc", "test_acc"])
        csv_w.writeheader()
        log_msg("dltrain-007: Data augmentation hurts at low epochs")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        for with_aug, label in [(False, "no_aug"), (True, "with_aug")]:
            log_msg(f"\n--- {label} ---")
            train_ldr, test_ldr = get_loaders(with_aug)
            model = CNN().cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            for ep in range(EPOCHS):
                model.train(); total_loss, correct, n = 0, 0, 0
                for x, y in train_ldr:
                    x, y = x.cuda(), y.cuda(); opt.zero_grad()
                    with torch.amp.autocast("cuda"): loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    total_loss += loss.item() * x.size(0); correct += (model(x).argmax(1) == y).sum().item(); n += x.size(0)
                model.eval(); t_correct, t_n = 0, 0
                for x, y in test_ldr: x, y = x.cuda(), y.cuda(); t_correct += (model(x).argmax(1) == y).sum().item(); t_n += x.size(0)
                test_acc = t_correct / t_n
                log_msg(f"  ep {ep+1:>2}: loss={total_loss/n:.4f} train_acc={correct/n:.3f} test_acc={test_acc:.3f}")
                csv_w.writerow({"augmentation": label, "epoch": ep+1, "train_loss": round(total_loss/n,4), "train_acc": round(correct/n,4), "test_acc": round(test_acc,4)})
            del model, opt, scaler
        log_msg("\nDone.")

if __name__ == "__main__": main()
