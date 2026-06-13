"""Standalone visualization script — loads a saved model checkpoint and generates
explainability dashboards. No training, just inference.

Usage: python visualize.py --model model.pt --output-dir /content/cnn-explainer-output
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="/content/cnn-explainer-output/model.pt")
    p.add_argument("--output-dir", type=str, default="/content/cnn-explainer-output")
    p.add_argument("--num-explain", type=int, default=16)
    p.add_argument("--ig-steps", type=int, default=20)
    return p.parse_args()

CFG = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10

os.makedirs(os.path.join(CFG.output_dir, "pngs"), exist_ok=True)

eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
viz_tf = T.Compose([T.ToTensor()])


# ── Model ───────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, NUM_CLASSES)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
    @property
    def last_conv(self):
        return self.block3.conv


# ── Explainability ──────────────────────────────────────────────────────────────

def grad_cam(model, x, class_idx):
    activations = []
    gradients = []
    def fwd_hook(m, inp, out): activations.append(out)
    def bwd_hook(m, gi, go): gradients.append(go[0])

    target = model.last_conv
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    out = model(x)
    model.zero_grad()
    out[0, class_idx].backward()

    pooled = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (pooled * activations[0]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.squeeze().detach().cpu().numpy()


def saliency_map(model, x, class_idx):
    x_in = x.clone().detach().requires_grad_(True)
    out = model(x_in)
    model.zero_grad()
    out[0, class_idx].backward()
    sal = x_in.grad.abs().max(dim=1)[0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.squeeze().detach().cpu().numpy()


def integrated_gradients(model, x, class_idx, steps=20):
    baseline = torch.zeros_like(x)
    ig = torch.zeros_like(x)
    for alpha in np.linspace(0, 1, steps):
        interp = baseline + alpha * (x - baseline)
        interp = interp.clone().detach().requires_grad_(True)
        out = model(interp)
        model.zero_grad()
        out[0, class_idx].backward()
        ig = ig + interp.grad.detach()
    ig = (ig * (x - baseline)).abs().max(dim=1)[0]
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
    return ig.squeeze().detach().cpu().numpy()


# ── Visualization ───────────────────────────────────────────────────────────────

def save_explainer_dashboard(model, test_raw, num, output_dir):
    model.eval()
    n = min(num, len(test_raw))
    indices = np.random.choice(len(test_raw), n, replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for c, title in enumerate(["Input", "Grad-CAM", "Saliency Map", "Int. Gradients"]):
        axes[0][c].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        idx = int(idx)
        sample = test_raw[idx]
        img_pil = sample["img"]
        true_label = CLASS_NAMES[sample["label"]]

        img_viz = viz_tf(img_pil).unsqueeze(0).to(DEVICE)
        img_eval = eval_tf(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(img_eval)
            pred_id = out.argmax(1).item()

        pred_label = CLASS_NAMES[pred_id]
        correct = pred_id == sample["label"]
        title_color = "green" if correct else "red"

        # Column 0: Input
        img_np = img_viz.squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[row][0].imshow(img_np)
        axes[row][0].set_ylabel(f"T: {true_label}\nP: {pred_label}", fontsize=8, color=title_color)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])

        cam = grad_cam(model, img_eval, pred_id)
        sal = saliency_map(model, img_eval, pred_id)
        ig = integrated_gradients(model, img_eval, pred_id, steps=CFG.ig_steps)

        axes[row][1].imshow(img_np)
        axes[row][1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row][1].set_xticks([])
        axes[row][1].set_yticks([])

        axes[row][2].imshow(sal, cmap="hot")
        axes[row][2].set_xticks([])
        axes[row][2].set_yticks([])

        axes[row][3].imshow(ig, cmap="hot")
        axes[row][3].set_xticks([])
        axes[row][3].set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "explainer_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved explainer dashboard → {path}")


def save_feature_maps(model, test_raw, output_dir):
    model.eval()
    indices = np.random.choice(len(test_raw), 64, replace=False)
    imgs = []
    for idx in indices:
        sample = test_raw[int(idx)]
        imgs.append(eval_tf(sample["img"]))
    x = torch.stack(imgs).to(DEVICE)

    activations = {}
    def make_hook(name):
        def hook(m, inp, out): activations[name] = out
        return hook

    hooks = [
        model.block1.conv.register_forward_hook(make_hook("block1")),
        model.block2.conv.register_forward_hook(make_hook("block2")),
        model.block3.conv.register_forward_hook(make_hook("block3")),
    ]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    fig, all_axes = plt.subplots(3, 8, figsize=(16, 7))
    for layer_idx, (name, acts) in enumerate(activations.items()):
        n_filters = acts.shape[1]
        selected = np.random.choice(n_filters, min(8, n_filters), replace=False)
        for col, fidx in enumerate(selected):
            ax = all_axes[layer_idx][col]
            filter_acts = acts[:, fidx, :, :].mean(dim=(1, 2))
            best_img_idx = filter_acts.argmax().item()
            best_img = imgs[best_img_idx]
            img_np = best_img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{name}\nch={acts.shape[1]}", fontsize=9)
            ax.set_title(f"F#{fidx}", fontsize=8)

    plt.suptitle("Feature Maps — top-activating image per filter", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "pngs", "feature_maps.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved feature maps → {path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading model from {CFG.model}...")
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(CFG.model, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Model loaded. Device: {DEVICE}  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading CIFAR-10 test set...")
    ds = load_dataset("uoft-cs/cifar10")
    test_raw = ds["test"]
    print(f"Test set: {len(test_raw)} images")

    print("Generating explainer dashboard...")
    save_explainer_dashboard(model, test_raw, CFG.num_explain, CFG.output_dir)

    print("Generating feature maps...")
    save_feature_maps(model, test_raw, CFG.output_dir)

    print("Done!")

if __name__ == "__main__":
    main()
