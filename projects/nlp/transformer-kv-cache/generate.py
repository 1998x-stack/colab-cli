"""Generate text and compare inference latency: with vs without KV cache.

Demonstrates KV cache speedup by measuring per-token latency at each step
and plotting the cumulative time difference.

Usage:
  python generate.py --checkpoint output/checkpoints/weights_epoch10.pt
  python generate.py --checkpoint output/checkpoints/weights_epoch10.pt --prompt "ROMEO:" --max_tokens 200
"""
import argparse
import os
import time
import urllib.request

import torch
import torch.nn.functional as F

from config import TransformerConfig
from model import GPT
from kv_cache import KVCache


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    path = "/tmp/shakespeare/input.txt"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path) as f:
        return f.read()


def get_tokenizer(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="First Citizen:")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    # Load checkpoint first to infer architecture
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ckpt_vocab_size = ckpt["model_state"]["token_embed.weight"].shape[0]
    ckpt_block_size = ckpt["model_state"]["pos_embed.weight"].shape[0]
    print(f"[generate] Checkpoint: vocab_size={ckpt_vocab_size}, block_size={ckpt_block_size}")

    # Load text and build tokenizer (for encoding/decoding only)
    text = load_text()
    stoi, itos, vocab_size = get_tokenizer(text)
    if ckpt_vocab_size < vocab_size:
        print(f"[generate] Warning: checkpoint was trained on a subset ({ckpt_vocab_size} of {vocab_size} chars). Using checkpoint's vocab.")
    print(f"[generate] Full text vocab: {vocab_size}")

    # Build model matching checkpoint architecture
    config = TransformerConfig(
        vocab_size=ckpt_vocab_size,
        block_size=args.block_size if args.block_size is not None else ckpt_block_size,
    )
    model = GPT(config, use_cache=True).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[generate] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Encode prompt
    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long).to(device)
    print(f"[generate] Prompt: '{args.prompt}' ({prompt_ids.shape[1]} tokens)")

    # --- Generate WITH KV cache ---
    print("\n" + "=" * 60)
    print("Generating WITH KV Cache")
    print("=" * 60)

    gen_with, lats_with = model.generate(
        prompt_ids.clone(), max_new_tokens=args.max_tokens,
        temperature=args.temperature, use_cache=True,
    )
    output_with = "".join(itos.get(i, "?") for i in gen_with[0].tolist())
    print(output_with)
    print(f"\nLatency: {sum(lats_with):.0f}ms total, {sum(lats_with)/len(lats_with):.1f}ms avg/token")

    # Reset and generate WITHOUT KV cache
    print("\n" + "=" * 60)
    print("Generating WITHOUT KV Cache (recomputing full attention)")
    print("=" * 60)

    gen_without, lats_without = model.generate(
        prompt_ids.clone(), max_new_tokens=args.max_tokens,
        temperature=args.temperature, use_cache=False,
    )
    output_without = "".join(itos.get(i, "?") for i in gen_without[0].tolist())
    print(output_without)
    print(f"\nLatency: {sum(lats_without):.0f}ms total, {sum(lats_without)/len(lats_without):.1f}ms avg/token")

    # --- Summary ---
    speedup = sum(lats_without) / max(sum(lats_with), 1)
    print("\n" + "=" * 60)
    print("KV CACHE SPEEDUP SUMMARY")
    print("=" * 60)
    print(f"  With cache:    {sum(lats_with):.0f}ms total ({sum(lats_with)/len(lats_with):.1f}ms/step)")
    print(f"  Without cache: {sum(lats_without):.0f}ms total ({sum(lats_without)/len(lats_without):.1f}ms/step)")
    print(f"  Speedup:       {speedup:.1f}x")
    print(f"  Tokens generated: {args.max_tokens}")

    # --- Plot latency comparison ---
    _make_speedup_chart(lats_with, lats_without, args.checkpoint)


def _make_speedup_chart(lats_with, lats_without, ckpt_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ckpt_dir = os.path.dirname(ckpt_path)
    png_dir = os.path.join(ckpt_dir, "..", "pngs")
    os.makedirs(png_dir, exist_ok=True)

    steps = np.arange(1, len(lats_with) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step latency
    ax1.plot(steps, lats_with, "b-", label="With KV Cache", linewidth=2, alpha=0.8)
    ax1.plot(steps, lats_without, "r-", label="Without KV Cache", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Generation Step")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-Step Latency: KV Cache vs No Cache")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative latency
    cum_with = np.cumsum(lats_with)
    cum_without = np.cumsum(lats_without)
    ax2.plot(steps, cum_with, "b-", label="With KV Cache", linewidth=2)
    ax2.plot(steps, cum_without, "r-", label="Without KV Cache", linewidth=2)
    ax2.set_xlabel("Generation Step")
    ax2.set_ylabel("Cumulative Latency (ms)")
    ax2.set_title("Cumulative Time: O(L) vs O(L^2)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "kv_cache_speedup.png"), dpi=150)
    plt.close(fig)
    print(f"[generate] Speedup chart saved to {png_dir}/kv_cache_speedup.png")


if __name__ == "__main__":
    main()
