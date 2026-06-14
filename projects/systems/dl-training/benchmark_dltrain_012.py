"""dltrain-012: AdamW weight decay on bias/BN parameters — harmful.

Applying weight_decay to bias terms and BN gamma/beta harms training.
Standard: exclude bias & norm params from weight decay via param groups.
"""
import torch, torch.nn.functional as F, os, csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-012")
EPOCHS = 15; BATCH = 128

def setup():
    for sub in ["logs", "pngs"]: Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)

def get_data():
    import torchvision.transforms as T, torchvision.datasets as D
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    test_ds = D.CIFAR10(root="/content/data", train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True),
            torch.utils.data.DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True))

class CNN_BN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1); x = x.flatten(1)
        return self.fc(x)

def split_params(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

def main():
    setup(); train_ldr, test_ldr = get_data()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["method", "epoch", "train_loss", "test_acc"])
        csv_w.writeheader()
        log_msg("dltrain-012: weight decay on bias/BN parameters")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        for method in ["wd_on_all", "wd_exclude_bias_bn"]:
            log_msg(f"\n--- {method} ---")
            model = CNN_BN().cuda()
            if method == "wd_exclude_bias_bn":
                decay, no_decay = split_params(model)
                opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
            else:
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda")
            for ep in range(EPOCHS):
                model.train(); total_loss, n = 0, 0
                for x, y in train_ldr:
                    x, y = x.cuda(), y.cuda(); opt.zero_grad()
                    with torch.amp.autocast("cuda"): loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    total_loss += loss.item() * x.size(0); n += x.size(0)
                model.eval(); t_correct, t_n = 0, 0
                for x, y in test_ldr: x, y = x.cuda(), y.cuda(); t_correct += (model(x).argmax(1) == y).sum().item(); t_n += x.size(0)
                log_msg(f"  ep {ep+1:>2}: loss={total_loss/n:.4f} test_acc={t_correct/t_n:.3f}")
                csv_w.writerow({"method": method, "epoch": ep+1, "train_loss": round(total_loss/n,4), "test_acc": round(t_correct/t_n,4)})
        log_msg("\nDone.")

if __name__ == "__main__": main()
