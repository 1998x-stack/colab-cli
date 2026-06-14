"""dltrain-010: SWA (Stochastic Weight Averaging) — free 1-2% accuracy.

Average checkpoints from last ~25% of training via torch.optim.swa_utils.
"""
import torch, torch.nn.functional as F, os, csv, copy
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-010")
EPOCHS = 20; BATCH = 128; SWA_START = 14

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
def update_swa(swa_model, model, n_avg):
    for swa_p, p in zip(swa_model.parameters(), model.parameters()):
        swa_p.data = (swa_p.data * (n_avg - 1) + p.data) / n_avg

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct, n = 0, 0
    for x, y in loader: x, y = x.cuda(), y.cuda(); correct += (model(x).argmax(1) == y).sum().item(); n += x.size(0)
    return correct / n

def main():
    setup(); train_ldr, test_ldr = get_data()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["method", "epoch", "test_acc"])
        csv_w.writeheader()
        log_msg("dltrain-010: SWA (Stochastic Weight Averaging)")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        for use_swa, label in [(False, "no_swa"), (True, "with_swa")]:
            log_msg(f"\n--- {label} ---")
            model = CNN().cuda()
            swa_model = copy.deepcopy(model) if use_swa else None
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            n_swa = 0
            for ep in range(EPOCHS):
                model.train()
                for x, y in train_ldr:
                    x, y = x.cuda(), y.cuda(); opt.zero_grad()
                    with torch.amp.autocast("cuda"): loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                if use_swa and ep >= SWA_START:
                    n_swa += 1; update_swa(swa_model, model, n_swa)
                eval_model = swa_model if (use_swa and ep >= SWA_START) else model
                acc = evaluate(eval_model, test_ldr)
                swa_tag = " (SWA)" if (use_swa and ep >= SWA_START) else ""
                log_msg(f"  ep {ep+1:>2}: test_acc={acc:.3f}{swa_tag}")
                csv_w.writerow({"method": label, "epoch": ep+1, "test_acc": round(acc, 4)})
        log_msg("\nDone.")

if __name__ == "__main__": main()
