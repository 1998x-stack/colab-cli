"""Training loop for Transformer on IWSLT'17 De->En.

Usage:
    python train.py --exp_id baseline
    python train.py --exp_id baseline --resume /content/checkpoints/ckpt_epoch5.pt
"""
import argparse, json, math, os, sys, time, urllib.request, zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import sacrebleu

from model import build_transformer, Transformer
from checkpoint import save_checkpoint, load_checkpoint, ensure_checkpoint_dir


# --- Constants ---
# HF canonical repo is IWSLT/iwslt2017 (uppercase org). Lowercase redirects via 307
# which urllib doesn't follow, so use the canonical casing directly.
IWSLT_ZIP_URL = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/de/en/de-en.zip"
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]


def _parse_iwslt_line(line: str) -> str | None:
    """Extract text from an IWSLT tagged line. Skips meta tags, keeps plain text."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("<"):
        # Skip XML meta tags: <doc>, <url>, <keywords>, <speaker>, <talkid>, <title>, <description>
        # Also skip <seg> if present (some versions have them)
        if line.startswith("<seg"):
            part = line.split(">", 1)[1]
            text = part.rsplit("<", 1)[0]
            return text.strip()
        return None
    return line


def load_iwslt_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Download and load IWSLT'17 De-En from HF CDN ZIP file."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "de-en.zip")
    de_path = os.path.join(data_dir, "train.de")
    en_path = os.path.join(data_dir, "train.en")

    if not os.path.exists(de_path) or not os.path.exists(en_path):
        if not os.path.exists(zip_path):
            print(f"[data] Downloading IWSLT ZIP (~18MB)...")
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(IWSLT_ZIP_URL, zip_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"[data] Retry {attempt+1}/3: {e}")
                    time.sleep(2)

        print("[data] Extracting ZIP...")
        with zipfile.ZipFile(zip_path) as zf:
            # ZIP extracts to de-en/ directory with train.tags.de-en.{de,en}
            de_zip = "de-en/train.tags.de-en.de"
            en_zip = "de-en/train.tags.de-en.en"
            if de_zip not in zf.namelist():
                # Debug: list contents
                names = [n for n in zf.namelist() if "train" in n.lower()]
                raise RuntimeError(f"train.tags.de-en not found. Contents with 'train': {names}")

            with zf.open(de_zip) as src, open(de_path, "wb") as dst:
                dst.write(src.read())
            with zf.open(en_zip) as src, open(en_path, "wb") as dst:
                dst.write(src.read())

    # Parse XML-tagged files, extracting only <seg> lines (aligned)
    with open(de_path) as df, open(en_path) as ef:
        de_lines = df.readlines()
        en_lines = ef.readlines()

    pairs = []
    for de_line, en_line in zip(de_lines, en_lines):
        de_text = _parse_iwslt_line(de_line)
        en_text = _parse_iwslt_line(en_line)
        if de_text is not None and en_text is not None:
            if de_text and en_text:  # skip empty
                pairs.append((de_text, en_text))

    print(f"[data] Loaded {len(pairs)} sentence pairs")
    return pairs


# --- Tokenizer ---

def train_tokenizer(
    pairs: list[tuple[str, str]], vocab_size: int = 32000, save_path: str = "/content/tokenizer.json"
) -> Tokenizer:
    """Train a shared BPE tokenizer on concatenated source + target sentences."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    all_text = [de for de, _ in pairs] + [en for _, en in pairs]
    tokenizer.train_from_iterator(all_text, trainer)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tokenizer.save(save_path)

    return tokenizer


# --- Dataset ---

class TranslationDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        de, en = self.pairs[idx]
        src_ids = [SOS] + self.tokenizer.encode(de).ids[:self.max_len - 2] + [EOS]
        tgt_ids = [SOS] + self.tokenizer.encode(en).ids[:self.max_len - 2] + [EOS]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch: list, pad_idx: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# --- LR Scheduler (paper Sec 5.3) ---

class NoamScheduler:
    """lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))"""
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        arg1 = self._step ** (-0.5)
        arg2 = self._step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    def state_dict(self):
        return {"step": self._step, "rate": self._rate}

    def load_state_dict(self, state: dict):
        self._step = state["step"]
        self._rate = state["rate"]


# --- Beam Search ---

@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    beam_size: int,
    eos_idx: int,
    device: torch.device,
) -> list[int]:
    """Beam search decode. src is (1, src_len). Returns token list (without SOS/EOS)."""
    model.eval()
    enc_out = model.encoder(src, src_mask)  # (1, src_len, d_model)
    # Expand to beam_size: (beam_size, src_len, d_model)
    enc_out_b = enc_out.expand(beam_size, -1, -1)
    src_mask_b = src_mask.expand(beam_size, -1, -1, -1) if src_mask is not None else None

    # Each beam: (beam_size, seq_len_so_far)
    sequences = torch.full((beam_size, 1), SOS, dtype=torch.long, device=device)
    scores = torch.zeros(beam_size, device=device)
    finished = torch.zeros(beam_size, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        if finished.all():
            break

        tgt = sequences
        tgt_mask = (Transformer.create_padding_mask(PAD, tgt) |
                    Transformer.create_causal_mask(tgt.size(1), device))

        dec_out = model.decoder(tgt, enc_out_b, src_mask_b, tgt_mask)
        logits = model.out_proj(dec_out[:, -1, :])  # (beam_size, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # For finished beams, only allow EOS
        log_probs[finished] = float("-inf")
        log_probs[finished, eos_idx] = 0.0

        # (beam_size, vocab) candidates
        cand_scores = scores.unsqueeze(1) + log_probs  # (beam_size, vocab)
        cand_scores_flat = cand_scores.view(-1)
        top_scores, top_idx = torch.topk(cand_scores_flat, beam_size)

        beam_idx = top_idx // log_probs.size(1)
        token_idx = top_idx % log_probs.size(1)

        new_sequences = torch.zeros(beam_size, step + 2, dtype=torch.long, device=device)
        new_scores = torch.zeros(beam_size, device=device)
        new_finished = torch.zeros(beam_size, dtype=torch.bool, device=device)

        for i in range(beam_size):
            src_beam = beam_idx[i]
            new_sequences[i, :step+1] = sequences[src_beam]
            new_sequences[i, step+1] = token_idx[i]
            new_scores[i] = top_scores[i]
            new_finished[i] = finished[src_beam] | (token_idx[i] == eos_idx)

        sequences = new_sequences
        scores = new_scores
        finished = new_finished

    best_idx = scores.argmax().item()
    tokens = sequences[best_idx].tolist()
    if eos_idx in tokens:
        tokens = tokens[:tokens.index(eos_idx)]
    return tokens[1:]  # skip SOS



# --- BLEU Evaluation ---

@torch.no_grad()
def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 128,
) -> float:
    """Compute sacreBLEU on validation set."""
    model.eval()
    hypotheses = []
    references = []

    for src, tgt in dataloader:
        src = src.to(device)
        src_mask = Transformer.create_padding_mask(PAD, src)

        for i in range(src.size(0)):
            pred_tokens = beam_search(
                model, src[i:i+1], src_mask[i:i+1], max_len, beam_size, EOS, device
            )
            hyp = tokenizer.decode(pred_tokens)
            ref_tokens = [t for t in tgt[i].tolist() if t not in (PAD, SOS, EOS)]
            ref = tokenizer.decode(ref_tokens)
            hypotheses.append(hyp)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# --- Validation Loss ---

@torch.no_grad()
def _compute_val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = Transformer.create_padding_mask(PAD, src)
        tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                    Transformer.create_causal_mask(tgt_in.size(1), device))
        logits = model(src, tgt_in, src_mask, tgt_mask)
        total += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        n += 1
    return total / max(n, 1)


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", required=True, choices=["baseline", "fixed_pe", "heads_1"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir", default="/content/iwslt_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--output_dir", default="/content")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}, Exp: {args.exp_id}")

    # --- Data ---
    pairs = load_iwslt_pairs(args.data_dir)

    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
        print(f"[train] Loaded tokenizer from {tok_path}")
    else:
        tokenizer = train_tokenizer(pairs, save_path=tok_path)
        print(f"[train] Trained tokenizer, vocab={tokenizer.get_vocab_size()}")

    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] Vocab size: {vocab_size}, Pairs: {len(pairs)}")

    # Train/val split
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = TranslationDataset(train_pairs, tokenizer, args.max_len)
    val_ds = TranslationDataset(val_pairs, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # --- Model ---
    model = build_transformer(args.exp_id, vocab_size).to(device)
    print(f"[train] Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=model.d_model, warmup_steps=4000)

    # Resume state
    start_epoch = 0
    tokens_processed = 0
    wall_time_s = 0.0
    metrics_file = os.path.join(args.output_dir, "metrics.jsonl")

    if args.resume:
        opt_state, sched_state, start_epoch, prev_metrics, _ = load_checkpoint(
            args.resume, model, device
        )
        optimizer.load_state_dict(opt_state)
        if sched_state:
            scheduler.load_state_dict(sched_state)
        tokens_processed = prev_metrics["tokens_processed"]
        wall_time_s = prev_metrics["wall_time_s"]
        print(f"[train] Resumed from epoch {start_epoch} (loss={prev_metrics['train_loss']:.3f}, bleu={prev_metrics['bleu']:.1f})")

    if not args.resume:
        # Create empty metrics file
        with open(metrics_file, "w") as f:
            pass

    # --- Config ---
    config = {
        "exp_id": args.exp_id, "vocab_size": vocab_size, "d_model": model.d_model,
        "n_heads": model.encoder.layers[0].self_attn.n_heads,
        "batch_size": args.batch_size, "max_len": args.max_len,
        "beam_size": args.beam_size, "train_pairs": len(train_pairs),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    ckpt_dir = ensure_checkpoint_dir(args.output_dir)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.0)
    t0 = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = Transformer.create_padding_mask(PAD, src)
            tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                        Transformer.create_causal_mask(tgt_in.size(1), device))

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            tokens_processed += src.numel() + tgt.numel()

        train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        val_loss = _compute_val_loss(model, val_loader, criterion, device)
        bleu = evaluate(model, val_loader, tokenizer, device, beam_size=args.beam_size, max_len=args.max_len)

        epoch_time = time.time() - t0
        wall_time_s += epoch_time
        t0 = time.time()

        # --- Log ---
        lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "bleu": round(bleu, 1),
            "lr": round(lr, 8), "tokens_processed": tokens_processed,
            "wall_time_s": round(wall_time_s, 1),
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"[train] Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | "
              f"BLEU={bleu:.1f} | lr={lr:.6f} | time={wall_time_s/60:.1f}m")

        # --- Checkpoint ---
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch,
                        train_loss, val_loss, bleu, tokens_processed, wall_time_s, config)
        # Remove older checkpoint (keep last 2)
        if epoch > 2:
            old = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch-2}.pt")
            if os.path.exists(old):
                os.remove(old)

        t0 = time.time()

    print(f"[train] Done. Total time: {wall_time_s/60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
