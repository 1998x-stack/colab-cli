# cnn-explainer gotchas

## Verified working

- `uoft-cs/cifar10` loads correctly on Colab (tested in cnn-cifar10 project)
- Grad-CAM hook pattern: `register_forward_hook` + `register_full_backward_hook` on `model.last_conv`
- `F.interpolate` upsampling CAM to input size works on GPU

## Design decisions

- **Saliency = max across RGB channels** (not sum or L2 norm). Max preserves the most salient channel per pixel, avoids washout from inactive channels.
- **Integrated Gradients baseline = zero image** (black). Standard for CIFAR-10 where zero = "no information." Alternatives (blurred, random) don't add value for 32×32 images.
- **`weights_only=True` on torch.load** — required by PyTorch 2.6+ security defaults.
- **Feature maps show top-1 activating image per filter** — simpler than top-9 grid and avoids clutter. Use `mean(dim=(1,2))` pooling for activation score.
- **No Guided Backprop in dashboard** — Integrated Gradients provides cleaner attribution without the "backward ReLU" hack that introduces artifacts.

## If things break

### CUDA OOM during explainability
The dashboard processes 16 images sequentially (not batched), so VRAM should be fine on T4. If OOM: reduce `--num-explain` to 8.

### Grad-CAM produces uniform (all-zero) heatmap
Usually means the model is classifying based on texture outside the receptive field of the last conv layer. Check `model.last_conv` is actually the right layer (should be block3.conv).

### HuggingFace dataset download fails
`uoft-cs/cifar10` is public (no auth). If it fails, try the alternate identifier `cifar10` or download directly:
```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
