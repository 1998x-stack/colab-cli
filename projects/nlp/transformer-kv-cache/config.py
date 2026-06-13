"""All hyperparameters for the transformer + KV cache demo.

Everything lives in one dataclass so components can be imported independently
and receive a config object. CLI overrides any field via argparse.
"""
import argparse
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # --- Vocabulary ---
    vocab_size: int = 65  # tiny Shakespeare has ~65 unique chars
    pad_token_id: int = 0  # reserve 0 for padding (unused in char LM, but safe)

    # --- Architecture ---
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    block_size: int = 256  # max context length
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 64
    max_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # --- Logging ---
    log_interval: int = 50   # steps between log lines
    eval_interval: int = 200  # steps between eval runs
    chart_interval: int = 1  # epochs between chart overwrites

    # --- Output ---
    output_dir: str = "output"

    # --- Runtime ---
    device: str = "cpu"
    compile: bool = False  # torch.compile (PyTorch >= 2.0)

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_head


def parse_args() -> TransformerConfig:
    """Parse CLI arguments and return a populated config.

    Any config field can be overridden, e.g.:
        python train.py --n_layer 6 --d_model 512 --lr 1e-4 --device cuda
    """
    parser = argparse.ArgumentParser(
        description="Train a char-level GPT with KV cache support"
    )
    config = TransformerConfig()

    # Add all dataclass fields as CLI arguments
    for field_name, field_def in TransformerConfig.__dataclass_fields__.items():
        if field_name == "d_k":
            continue  # computed property, not a real field
        field_type = field_def.type
        default = getattr(config, field_name)

        if field_type is bool:
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
            parser.add_argument(f"--no-{field_name}", action="store_false", dest=field_name)
        elif field_type is int:
            parser.add_argument(f"--{field_name}", type=int, default=default)
        elif field_type is float:
            parser.add_argument(f"--{field_name}", type=float, default=default)
        elif field_type is str:
            parser.add_argument(f"--{field_name}", type=str, default=default)

    args = parser.parse_args()
    for field_name in TransformerConfig.__dataclass_fields__:
        if field_name == "d_k":
            continue
        setattr(config, field_name, getattr(args, field_name))

    return config
