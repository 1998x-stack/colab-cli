"""LoRA SFT fine-tuning of Qwen3-0.6B on tokenized text2sql data.

Usage:
    python train.py --data_path data/train.pt --output_dir lora_weights/
"""
import argparse
import csv
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-0.6B"
LOG_DIR = "logs"


class TensorDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad to longest in batch."""
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = ex["input_ids"].shape[0]
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        labels[i, :n] = ex["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def setup_logging(output_dir):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "train.log")
    metrics_path = os.path.join(LOG_DIR, "metrics.csv")
    log_fh = open(log_path, "w")
    log_fh.write(f"[{time.strftime('%H:%M:%S')}] Training started\n")
    log_fh.flush()

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr", "elapsed_s"])

    return log_fh, metrics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="lora_weights")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    examples = torch.load(args.data_path, weights_only=False)
    dataset = TensorDataset(examples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

    log_fh, metrics_path = setup_logging(args.output_dir)
    step = 0
    t0 = time.time()

    for epoch in range(1):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                train_loss = loss.item() * args.grad_accum

                line = f"[{time.strftime('%H:%M:%S')}] step {step} | loss={train_loss:.4f} | lr={lr:.2e} | elapsed={elapsed:.0f}s"
                print(line)
                log_fh.write(line + "\n")
                log_fh.flush()

                with open(metrics_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([step, train_loss, lr, elapsed])

                if args.max_steps > 0 and step >= args.max_steps:
                    break

        if args.max_steps > 0 and step >= args.max_steps:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    total_elapsed = time.time() - t0
    line = f"[{time.strftime('%H:%M:%S')}] Training complete | steps={step} | total_time={total_elapsed:.0f}s"
    print(line)
    log_fh.write(line + "\n")
    log_fh.close()


if __name__ == "__main__":
    main()
