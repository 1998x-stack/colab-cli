"""Multi30k EN→DE dataset for seq2seq training on Colab T4.

Downloads raw data from multi30k GitHub, trains BPE tokenizer.
"""
import gzip
import os
import time
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]

MULTI30K_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
SPLITS = {
    "train": ("train.de.gz", "train.en.gz"),
    "val":   ("val.de.gz", "val.en.gz"),
    "test":  ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz"),
}


def _download_gz(url: str, dest: str):
    """Download .gz file and decompress to dest."""
    gz_path = dest + ".gz"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, gz_path)
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"[data] Retry {attempt+1}/3: {e}")
            time.sleep(2)

    with gzip.open(gz_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    os.remove(gz_path)


def load_multi30k(data_dir: str, reverse_src: bool = True) -> dict:
    """Download Multi30k EN→DE and return sentence pairs.

    Args:
        data_dir: cache directory for raw files
        reverse_src: if True, reverse source word order (paper's key insight)

    Returns:
        dict with keys 'train', 'val', 'test' → list of (src_str, tgt_str)
    """
    os.makedirs(data_dir, exist_ok=True)
    pairs = {}

    for split, (de_file, en_file) in SPLITS.items():
        de_path = os.path.join(data_dir, de_file.replace(".gz", ""))
        en_path = os.path.join(data_dir, en_file.replace(".gz", ""))

        if not os.path.exists(de_path):
            url_de = f"{MULTI30K_BASE}/{de_file}"
            url_en = f"{MULTI30K_BASE}/{en_file}"
            print(f"[data] Downloading {split} split...")
            _download_gz(url_de, de_path)
            _download_gz(url_en, en_path)

        with open(de_path) as df, open(en_path) as ef:
            de_lines = [l.strip() for l in df if l.strip()]
            en_lines = [l.strip() for l in ef if l.strip()]

        split_pairs = list(zip(en_lines, de_lines))  # EN→DE: src=en, tgt=de

        if reverse_src:
            split_pairs = [(" ".join(s.split()[::-1]), t) for s, t in split_pairs]

        pairs[split] = split_pairs
        print(f"[data] {split}: {len(split_pairs)} pairs (reverse_src={reverse_src})")

    return pairs


class TranslationDataset(Dataset):
    """Tokenized translation pairs."""

    def __init__(self, pairs: list[tuple[str, str]], src_tokenizer, tgt_tokenizer,
                 src_max_len: int = 80, tgt_max_len: int = 80):
        self.samples = []
        for src, tgt in pairs:
            src_ids = src_tokenizer.encode(src).ids[:src_max_len]
            tgt_ids = tgt_tokenizer.encode(tgt).ids[:tgt_max_len - 2]
            # Add <SOS> and <EOS> to target
            tgt_ids = [SOS] + tgt_ids + [EOS]
            self.samples.append((torch.tensor(src_ids, dtype=torch.long),
                                 torch.tensor(tgt_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad src and tgt to max length in batch."""
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True,
                                                  padding_value=PAD)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True,
                                                  padding_value=PAD)
    return src_padded, tgt_padded


def build_tokenizer(pairs: list[tuple[str, str]], vocab_size: int = 8000,
                    lang: str = "src") -> Tokenizer:
    """Train a BPE tokenizer on the given sentence pairs.

    Args:
        pairs: list of (src, tgt) strings
        vocab_size: BPE vocabulary size
        lang: 'src' or 'tgt' — which side to train on
    """
    idx = 0 if lang == "src" else 1
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    def text_iter():
        for pair in pairs:
            yield pair[idx]

    tokenizer.train_from_iterator(text_iter(), trainer)
    print(f"[tokenizer] {lang}: vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


def build_dataloaders(pairs: dict, src_tokenizer, tgt_tokenizer,
                      batch_size: int = 64, src_max_len: int = 80,
                      tgt_max_len: int = 80) -> dict:
    """Build DataLoader for each split."""
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = TranslationDataset(pairs[split], src_tokenizer, tgt_tokenizer,
                                src_max_len, tgt_max_len)
        shuffle = (split == "train")
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    collate_fn=collate_fn, pin_memory=True)
    return loaders
