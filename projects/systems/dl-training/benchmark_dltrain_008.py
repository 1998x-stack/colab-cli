"""dltrain-008: bias=False before BatchNorm — the dead parameter tax.

Linear/Conv bias immediately subtracted by BN. Always set bias=False before BN.
"""
import torch, torch.nn.functional as F, os, csv
from pathlib import Path

OUT_DIR = os.environ.get("OUT_DIR", "/content/dl-training-output/dltrain-008")
EPOCHS = 10; BATCH = 128

def setup():
    for sub in ["logs", "pngs"]: Path(OUT_DIR, sub).mkdir(parents=True, exist_ok=True)

def get_data():
    import torchvision.transforms as T, torchvision.datasets as D
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

def make_cnn(bias_before_bn):
    layers = []
    for _ in range(3):
        layers += [torch.nn.Conv2d(32, 32, 3, padding=1, bias=bias_before_bn),
                   torch.nn.BatchNorm2d(32), torch.nn.ReLU()]
    first = torch.nn.Conv2d(3, 32, 3, padding=1, bias=bias_before_bn)
    return torch.nn.Sequential(first, torch.nn.BatchNorm2d(32), torch.nn.ReLU(), *layers,
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 10))

def count_params(model): return sum(p.numel() for p in model.parameters())

def main():
    setup(); loader = get_data()
    log_path = os.path.join(OUT_DIR, "logs", "benchmark.log")
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(log_path, "w") as lf, open(csv_path, "w", newline="") as cf:
        def log_msg(msg): print(msg, flush=True); lf.write(msg + "\n")
        csv_w = csv.DictWriter(cf, fieldnames=["bias_config", "epoch", "train_loss", "train_acc", "param_count"])
        csv_w.writeheader()
        log_msg("dltrain-008: bias=False before BatchNorm")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
        for bias_first, label in [(True, "bias_before_bn"), (False, "no_bias_before_bn")]:
            model = make_cnn(bias_first).cuda()
            n_params = count_params(model)
            log_msg(f"\n--- {label} ({n_params:,} params) ---")
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            for ep in range(EPOCHS):
                model.train(); total_loss, correct, n = 0, 0, 0
                for x, y in loader:
                    x, y = x.cuda(), y.cuda(); opt.zero_grad()
                    with torch.amp.autocast("cuda"): loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    total_loss += loss.item() * x.size(0); correct += (model(x).argmax(1) == y).sum().item(); n += x.size(0)
                log_msg(f"  ep {ep+1:>2}: loss={total_loss/n:.4f} acc={correct/n:.3f}")
                csv_w.writerow({"bias_config": label, "epoch": ep+1, "train_loss": round(total_loss/n,4), "train_acc": round(correct/n,4), "param_count": n_params})
        log_msg(f"\nDead params wasted: {count_params(make_cnn(True)) - count_params(make_cnn(False)):,}")
        log_msg("\nDone.")

if __name__ == "__main__": main()
