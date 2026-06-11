"""QLoRA fine-tuning of Qwen2.5-7B-Instruct on filtered s1K data.

Usage:
    python train.py --data s1k_filtered.jsonl
    python train.py --data s1k_filtered.jsonl --output_dir /content/s1-t4/checkpoints --resume /path/to/adapter
"""
import argparse, json, os, sys, time, logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from torch.utils.data import Dataset


# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ASSISTANT_MARKER = "<|im_start|>assistant"
LOG_DIR = "/content/s1-t4/logs"
RESULTS_DIR = "/content/s1-t4/results"
HEARTBEAT_PATH = "/content/s1-t4/heartbeat.json"
TRAIN_LOSS_PATH = os.path.join(RESULTS_DIR, "train_loss.jsonl")


# --- Logging setup ---

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure structured logging to file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("s1-t4-train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler with timestamps
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(sh)

    return logger


def write_heartbeat(status: str, step: int, loss: float | None):
    """Write structured heartbeat JSON for external monitoring."""
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    heartbeat = {
        "status": status,
        "step": step,
        "loss": loss,
        "timestamp": time.time(),
    }
    with open(HEARTBEAT_PATH, "w") as f:
        json.dump(heartbeat, f)


def write_train_loss(step: int, loss: float):
    """Append a loss record to the per-step loss JSONL file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TRAIN_LOSS_PATH, "a") as f:
        f.write(json.dumps({"step": step, "loss": loss}) + "\n")


# --- Dataset ---

class S1KDataset(Dataset):
    """Dataset for s1K JSONL with loss masking.

    Each line: {"text": "<|im_start|>user\\n...\\n<|im_start|>assistant\\n...", "question": "...", "solution": "..."}
    Tokens before '<|im_start|>assistant' are masked (label=-100) so loss
    is only computed on think+answer tokens.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and parse JSONL
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        # Find the assistant marker token IDs
        self.assistant_ids = tokenizer.encode(
            ASSISTANT_MARKER, add_special_tokens=False
        )
        self.assistant_len = len(self.assistant_ids)

        print(f"[dataset] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        # Create labels: mask everything before (and including) the assistant marker
        labels = input_ids.copy()
        self._mask_prefix(labels, input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _mask_prefix(self, labels, input_ids):
        """Set labels to -100 for tokens up to and including '<|im_start|>assistant'."""
        # Search for the assistant marker token sequence in input_ids
        pos = self._find_subsequence(input_ids, self.assistant_ids)
        if pos is not None:
            mask_end = pos + self.assistant_len
            for i in range(mask_end):
                labels[i] = -100
        else:
            # Fallback: mask everything (no assistant marker found)
            for i in range(len(labels)):
                labels[i] = -100

    @staticmethod
    def _find_subsequence(seq, sub):
        """Find the start index of subsequence `sub` in sequence `seq`. Returns None if not found."""
        if len(sub) == 0:
            return None
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i + len(sub)] == sub:
                return i
        return None


# --- Logging Callback ---

class TrainLogCallback(TrainerCallback):
    """Custom callback that writes structured logs, heartbeat, and loss tracking."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss")

        # Write heartbeat
        write_heartbeat("training", step, loss)

        # Write loss trace
        if loss is not None:
            write_train_loss(step, loss)

        # Structured log
        self.logger.info(
            f"step={step} "
            + " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in logs.items())
        )

    def on_train_begin(self, args, state, control, **kwargs):
        write_heartbeat("started", 0, None)
        self.logger.info("Training started")

    def on_train_end(self, args, state, control, **kwargs):
        write_heartbeat("completed", state.global_step, None)
        self.logger.info(f"Training completed at step {state.global_step}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-7B-Instruct on s1K"
    )
    parser.add_argument("--data", required=True, help="Path to JSONL training data")
    parser.add_argument("--output_dir", default="/content/s1-t4/checkpoints",
                        help="Output directory for checkpoints (default: /content/s1-t4/checkpoints)")
    parser.add_argument("--resume", default=None,
                        help="Path to adapter checkpoint for resume training")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Setup logging
    logger = setup_logging(LOG_DIR)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Model: {MODEL_NAME}")

    # --- Load tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load dataset ---
    logger.info(f"Loading dataset from {args.data}...")
    dataset = S1KDataset(args.data, tokenizer, max_length=args.max_seq_length)
    logger.info(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        logger.error("Empty dataset! Nothing to train.")
        sys.exit(1)

    # --- 4-bit quantization config ---
    logger.info("Configuring 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # --- Load base model ---
    logger.info(f"Loading {MODEL_NAME} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    logger.info("Base model loaded")

    # --- Resume from adapter ---
    if args.resume is not None:
        logger.info(f"Loading adapter from {args.resume}...")
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        logger.info("Adapter loaded, model is trainable")
    else:
        # --- LoRA config ---
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA applied")

    # Print trainable parameters
    model.print_trainable_parameters()

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=5,
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        run_name="s1-t4-qlora",
        ddp_find_unused_parameters=False,
    )

    # Compute effective batch size for logging
    effective_bs = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    logger.info(f"Effective batch size: {effective_bs}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[TrainLogCallback(logger)],
    )

    # --- Train ---
    logger.info("Starting training...")
    write_heartbeat("training", 0, None)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        write_heartbeat("interrupted", trainer.state.global_step, None)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        write_heartbeat("failed", trainer.state.global_step if hasattr(trainer, 'state') else 0, None)
        raise

    # --- Save final adapter ---
    final_dir = os.path.join(args.output_dir, "adapter_final")
    logger.info(f"Saving final adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Done!")

    # Write final heartbeat
    write_heartbeat("completed", trainer.state.global_step,
                    trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None)


if __name__ == "__main__":
    main()
