"""Checkpoint save/load helpers for training resume across Colab sessions."""
import torch
import os


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    epoch: int,
    train_loss: float,
    val_loss: float,
    bleu: float,
    tokens_processed: int,
    wall_time_s: float,
    config: dict,
):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bleu": bleu,
        "tokens_processed": tokens_processed,
        "wall_time_s": wall_time_s,
        "config": config,
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device):
    """Returns (optimizer_state, scheduler_state, epoch, metrics_dict, config). Caller restores optimizer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return (
        ckpt["optimizer_state"],
        ckpt.get("scheduler_state"),
        ckpt["epoch"],
        {
            "train_loss": ckpt.get("train_loss", float("inf")),
            "val_loss": ckpt.get("val_loss", float("inf")),
            "bleu": ckpt.get("bleu", 0.0),
            "tokens_processed": ckpt.get("tokens_processed", 0),
            "wall_time_s": ckpt.get("wall_time_s", 0.0),
        },
        ckpt.get("config", {}),
    )


def ensure_checkpoint_dir(base: str = "/content") -> str:
    path = os.path.join(base, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path
