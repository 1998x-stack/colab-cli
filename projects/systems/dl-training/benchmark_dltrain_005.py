"""dltrain-005: EMA of weights — free 0.5-1% accuracy.

Maintain shadow weights with EMA (beta=0.999). Use shadow at eval. ~5 lines, zero cost.
"""
import torch, torch.nn.functional as F, os, csv, copy
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-005")
EPOCHS = 15; BATCH = 128; EMA_BETA = 0.999

def setup():
    for sub in ["logs", "pngs"]: Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)

def get_data():
    import torchvision.transforms as T, torchvision.datasets as D
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    test_ds = D.CIFAR10(root="/content/data", train=False, download=True, transform=tf)
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

@torch.no_grad()
def update_ema(ema_model, model, beta):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1 - beta)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct, n = 0, 0
    for x, y in loader: x, y = x.cuda(), y.cuda(); correct += (model(x).argmax(1) == y).sum().item(); n += x.size(0)
    model.train(); return correct / n

def main():
    setup(); train_ldr, test_ldr = get_data()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["method", "epoch", "train_loss", "test_acc"])
        csv_w.writeheader()
        log_msg("dltrain-005: EMA of weights")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        for use_ema in [False, True]:
            label = "with_EMA" if use_ema else "no_EMA"
            log_msg(f"\n--- {label} ---")
            model = CNN().cuda()
            ema_model = copy.deepcopy(model) if use_ema else None
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            for ep in range(EPOCHS):
                model.train(); total_loss, n = 0, 0
                for x, y in train_ldr:
                    x, y = x.cuda(), y.cuda(); opt.zero_grad()
                    with torch.amp.autocast("cuda"): loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    total_loss += loss.item() * x.size(0); n += x.size(0)
                    if use_ema: update_ema(ema_model, model, EMA_BETA)
                eval_model = ema_model if use_ema else model
                acc = evaluate(eval_model, test_ldr)
                log_msg(f"  ep {ep+1:>2}: loss={total_loss/n:.4f} test_acc={acc:.3f}")
                csv_w.writerow({"method": label, "epoch": ep+1, "train_loss": round(total_loss/n,4), "test_acc": round(acc,4)})
        log_msg("\nDone.")

if __name__ == "__main__": main()
