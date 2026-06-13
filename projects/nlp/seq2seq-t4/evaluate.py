"""Evaluation: beam search BLEU on Multi30k test set.

Usage:
    python evaluate.py --ckpt checkpoints/weights_epoch10.pt
    python evaluate.py --ckpt checkpoints/weights_epoch10.pt --beam_width 5 --normal_order
"""
import argparse
import time

import torch
import sacrebleu

from model import build_seq2seq, PAD, SOS, EOS
from dataset import load_multi30k, build_tokenizer, build_dataloaders


def decode_tokens(token_ids: list[int], tokenizer) -> str:
    """Strip special tokens and decode to text."""
    clean = [t for t in token_ids if t not in (PAD, SOS, EOS)]
    return tokenizer.decode(clean)


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    pairs = load_multi30k(args.data_dir, reverse_src=not args.normal_order)
    src_tokenizer = build_tokenizer(pairs["train"], args.vocab_size, lang="src")
    tgt_tokenizer = build_tokenizer(pairs["train"], args.vocab_size, lang="tgt")
    loaders = build_dataloaders(pairs, src_tokenizer, tgt_tokenizer,
                                batch_size=1,  # beam search is sentence-at-a-time
                                src_max_len=args.max_len, tgt_max_len=args.max_len)

    # --- Model ---
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Try to get vocab sizes from checkpoint, fall back to tokenizer
    src_vocab = ckpt.get("src_vocab_size", src_tokenizer.get_vocab_size())
    tgt_vocab = ckpt.get("tgt_vocab_size", tgt_tokenizer.get_vocab_size())

    model = build_seq2seq(src_vocab, tgt_vocab,
                          embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, dropout=0.0, device=device)

    # Load weights (handle both full checkpoint and weights-only)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # --- Decode ---
    test_loader = loaders["test"]
    refs, hyps = [], []
    t_start = time.time()

    for idx, (src, tgt) in enumerate(test_loader):
        src = src.to(device)

        if args.beam_width > 1:
            results = model.beam_decode(src, beam_width=args.beam_width,
                                        max_len=args.max_len)
            hyp_tokens = results[0]  # best hypothesis
        else:
            outputs = model.greedy_decode(src, max_len=args.max_len)
            hyp_tokens = [t for t in outputs[0].tolist() if t not in (PAD, EOS)]

        ref_tokens = [t for t in tgt[0].tolist() if t not in (PAD, SOS, EOS)]

        ref_text = decode_tokens(ref_tokens, tgt_tokenizer)
        hyp_text = decode_tokens(hyp_tokens, tgt_tokenizer)

        if ref_text.strip() and hyp_text.strip():
            refs.append(ref_text)
            hyps.append(hyp_text)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"[{idx+1}/{len(test_loader)}] {elapsed:.0f}s — "
                  f"ref: {ref_text[:60]}...")
            print(f"  hyp: {hyp_text[:60]}...")

    # --- BLEU ---
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a")
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Beam width: {args.beam_width}")
    print(f"Reverse src: {not args.normal_order}")
    print(f"Sentences:  {len(hyps)}")
    print(f"BLEU:       {bleu.score:.1f}")
    print(f"Time:       {elapsed:.0f}s")
    print(f"{'='*60}")
    print(bleu)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Seq2Seq on Multi30k test")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--data_dir", type=str, default="/content/seq2seq-t4/data")
    parser.add_argument("--normal_order", action="store_true",
                        help="Use normal (non-reversed) source order")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
