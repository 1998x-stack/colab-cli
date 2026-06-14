# Text2SQL Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen3-0.6B on `b-mc2/sql-create-context` with LoRA SFT and execution-match eval, deployable to Colab T4 with cron monitoring.

**Architecture:** Four independent modules — `dataset.py` formats HF data into chat-template tensors, `train.py` applies LoRA and runs SFT, `evaluate.py` validates via in-memory SQLite execution-match, `launch.py` bootstraps everything on Colab. Cron watchtower via `fetch.sh` + `tar_outputs.py`.

**Tech Stack:** PyTorch, Transformers, PEFT (LoRA), datasets, sqlite3, argparse

---

### Task 1: Project scaffold

**Files:**
- Create: `projects/nlp/text2sql_finetune/__init__.py`
- Create: `tests/__init__.py` (if missing)

- [ ] **Step 1: Create directory and init files**

```bash
mkdir -p projects/nlp/text2sql_finetune
touch projects/nlp/text2sql_finetune/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add projects/nlp/text2sql_finetune/__init__.py tests/__init__.py
git commit -m "chore: scaffold text2sql_finetune project directory"
```

---

### Task 2: dataset.py — load, format, tokenize, save

**Files:**
- Create: `projects/nlp/text2sql_finetune/dataset.py`

- [ ] **Step 1: Write dataset.py**

```python
"""Load b-mc2/sql-create-context, format into Qwen3 chat template, save as .pt tensors.

Usage:
    python dataset.py --split train --max_examples 500 --output data/train.pt
    python dataset.py --split test --max_examples 100 --output data/test.pt
"""
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = "You are a SQL expert. Given a table schema and a question, write the correct SQL query."
MAX_LENGTH = 1024


def format_and_tokenize(tokenizer, context, question, answer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = [-100] * len(input_ids)

    prompt_len = len(prompt_tokens["input_ids"])
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test", "validation"])
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    dataset = load_dataset("b-mc2/sql-create-context", split=args.split)
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    examples = []
    for row in dataset:
        context = row.get("context", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not context or not question or not answer:
            continue
        tokenized = format_and_tokenize(tokenizer, context, question, answer)
        examples.append(tokenized)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(examples, args.output)
    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dataset.py locally with 5 examples to verify format**

```bash
cd projects/nlp/text2sql_finetune && python dataset.py --split train --max_examples 5 --output /tmp/test_data.pt
```

Expected: "Saved 5 examples to /tmp/test_data.pt"

- [ ] **Step 3: Verify the saved tensors have correct structure**

```bash
python -c "
import torch
data = torch.load('/tmp/test_data.pt')
ex = data[0]
print('input_ids shape:', ex['input_ids'].shape)
print('labels shape:', ex['labels'].shape)
print('Non-masked labels:', (ex['labels'] != -100).sum().item())
print('Last 5 labels:', ex['labels'][-5:].tolist())
"
```

Expected: Non-masked labels > 0, last labels not -100 (assistant tokens are unmasked).

- [ ] **Step 4: Commit**

```bash
git add projects/nlp/text2sql_finetune/dataset.py
git commit -m "feat: add dataset.py — load sql-create-context, format Qwen3 chat template, save .pt"
```

---

### Task 3: test_dataset.py — validate formatting, masking, truncation

**Files:**
- Create: `tests/test_text2sql_dataset.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for dataset.py — formatting, label masking, truncation."""
import torch
from projects.nlp.text2sql_finetune.dataset import format_and_tokenize


class FakeTokenizer:
    """Minimal tokenizer stub that returns predictable token IDs."""
    def __init__(self):
        self.im_start_id = 1
        self.im_end_id = 2
        self.newline_id = 3

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        """Return a predictable string based on the role content."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=False, **kwargs):
        """Tokenize by assigning each word a unique token ID."""
        words = text.split()
        if truncation and max_length:
            words = words[:max_length]
        return {
            "input_ids": list(range(100, 100 + len(words))),
            "attention_mask": [1] * len(words),
        }


def test_label_masking_only_assistant_tokens_unmasked():
    tokenizer = FakeTokenizer()
    result = format_and_tokenize(tokenizer, "CREATE TABLE t (x int)", "what is x?", "SELECT x FROM t")

    input_ids = result["input_ids"]
    labels = result["labels"]

    assert len(labels) == len(input_ids)
    assert labels[0] == -100, "first token (system) should be masked"
    assert labels[-1] != -100, "last token (assistant) should be unmasked"

    non_masked = (labels != -100).sum().item()
    assert non_masked > 0, "should have some unmasked tokens"
    assert non_masked < len(labels), "should have some masked tokens (not everything is assistant)"


def test_long_sequence_truncation():
    tokenizer = FakeTokenizer()
    long_context = "CREATE TABLE big ("
    for i in range(200):
        long_context += f"col{i} int, "
    long_context += "id int)"

    result = format_and_tokenize(tokenizer, long_context, "select all ids", "SELECT id FROM big")

    # With truncation at 80 tokens (set in FakeTokenizer via max_length override)
    assert len(result["input_ids"]) <= 1024, f"should truncate to 1024, got {len(result['input_ids'])}"


def test_truncated_example_preserves_label_alignment():
    """When truncated, labels and input_ids must stay same length."""
    tokenizer = FakeTokenizer()
    result = format_and_tokenize(tokenizer, "CREATE TABLE t (x int)", "what is x?", "SELECT x FROM t")
    assert len(result["input_ids"]) == len(result["labels"])
    assert len(result["input_ids"]) == len(result["attention_mask"])
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_text2sql_dataset.py -v
```

Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_text2sql_dataset.py
git commit -m "test: add dataset format, masking, and truncation tests"
```

---

### Task 4: train.py — LoRA SFT training loop

**Files:**
- Create: `projects/nlp/text2sql_finetune/train.py`

- [ ] **Step 1: Write train.py**

```python
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
    parser.add_argument("--max_seq_len", type=int, default=1024)
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
```

- [ ] **Step 2: Commit**

```bash
git add projects/nlp/text2sql_finetune/train.py
git commit -m "feat: add train.py — LoRA SFT training loop with logging"
```

---

### Task 5: test_train.py — forward pass and loss validation

**Files:**
- Create: `tests/test_text2sql_train.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for train.py — forward pass, loss behavior. Requires GPU or runs slow on CPU."""
import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Use a tiny 2-layer config for fast local testing
TINY_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "vocab_size": 1000,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def create_tiny_model():
    config = AutoConfig.for_model(**TINY_CONFIG)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    return model


def create_tiny_batch(batch_size=2, seq_len=64):
    """Returns a batch of tokenized data with some labels unmasked."""
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Last 20 tokens are "assistant" — unmasked
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[:, -20:] = input_ids[:, -20:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_forward_pass_no_nan():
    model = create_tiny_model()
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    batch = create_tiny_batch()
    outputs = model(**batch)
    loss = outputs.loss

    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"


def test_loss_decreases_after_step():
    model = create_tiny_model()
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    batch = create_tiny_batch()

    model.train()
    loss1 = model(**batch).loss
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss2 = model(**batch).loss
    assert loss2.item() < loss1.item(), f"Loss did not decrease: {loss2.item():.4f} >= {loss1.item():.4f}"
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_text2sql_train.py -v
```

Expected: 2 PASS (these use a tiny synthetic model, no GPU needed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_text2sql_train.py
git commit -m "test: add forward pass and loss convergence tests with tiny model"
```

---

### Task 6: evaluate.py — SQL generation + execution-match

**Files:**
- Create: `projects/nlp/text2sql_finetune/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

```python
"""Evaluate fine-tuned model via execution-match against in-memory SQLite databases.

Usage:
    python evaluate.py --data_path data/test.pt --lora_path lora_weights/ --output eval_report.json
"""
import argparse
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = "You are a SQL expert. Given a table schema and a question, write the correct SQL query."
SQL_TIMEOUT = 5
LOG_DIR = "logs"


def extract_sql(text):
    """Extract SQL from between <|im_start|>assistant and <|im_end|> tags."""
    pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: take everything after "assistant\n"
    idx = text.rfind("assistant\n")
    if idx != -1:
        return text[idx + len("assistant\n"):].strip()
    return text.strip()


def parse_create_table(sql):
    """Extract CREATE TABLE statements from context string."""
    pattern = r"CREATE\s+TABLE\s+[^;]+;"
    return re.findall(pattern, sql, re.IGNORECASE)


def execute_sql(create_tables, query):
    """Execute query against in-memory SQLite database built from CREATE TABLE statements.
    Returns (success, result_set_or_error).
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        for ct in create_tables:
            conn.execute(ct)
        conn.commit()
        cursor = conn.execute(query)
        rows = [tuple(row) for row in cursor.fetchall()]
        return True, rows
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def execute_with_timeout(create_tables, query, timeout=SQL_TIMEOUT):
    """Execute SQL with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(execute_sql, create_tables, query)
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            return False, "__TIMEOUT__"


def format_prompt(context, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
    ]
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output", default="eval_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    examples = torch.load(args.data_path, weights_only=False)

    os.makedirs(LOG_DIR, exist_ok=True)
    eval_log_path = os.path.join(LOG_DIR, "eval.log")
    eval_fh = open(eval_log_path, "w")

    results = []
    error_counts = {"syntax": 0, "timeout": 0, "wrong_result": 0, "other": 0}
    execution_matches = 0
    exact_matches = 0

    for i, ex in enumerate(examples):
        # Decode the prompt from input_ids (first non-masked part)
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        # Find where labels start (first non -100 position)
        label_start = 0
        for j in range(len(labels)):
            if labels[j] != -100:
                label_start = j
                break

        prompt_ids = input_ids[:label_start]
        gt_ids = [input_ids[j] for j in range(label_start, len(labels)) if labels[j] != -100]

        prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
        gt_sql = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

        # Extract context and question from prompt
        # prompt_text ends at "<|im_start|>assistant\n" — parse from the raw input_ids
        full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

        # Generate SQL
        model_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            generated = model.generate(
                **model_inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
        gen_sql = extract_sql(full_output)

        # Parse CREATE TABLE from the prompt
        # The context is in the original text before the generation
        context_text = tokenizer.decode(input_ids[:label_start].tolist(), skip_special_tokens=True)
        create_tables = parse_create_table(context_text)
        if not create_tables:
            create_tables = parse_create_table(full_text)

        # Compare
        exact_match = (gen_sql.strip().lower() == gt_sql.strip().lower())

        exec_match = False
        error_type = None
        if gen_sql:
            success, result = execute_with_timeout(create_tables, gen_sql)
            if success:
                gt_success, gt_result = execute_sql(create_tables, gt_sql)
                if gt_success:
                    exec_match = (result == gt_result)
                    if not exec_match:
                        error_type = "wrong_result"
                else:
                    error_type = "other"
            else:
                if result == "__TIMEOUT__":
                    error_type = "timeout"
                else:
                    error_type = "syntax"
        else:
            error_type = "syntax"

        if exec_match:
            execution_matches += 1
        if exact_match:
            exact_matches += 1
        if error_type:
            error_counts[error_type] += 1

        entry = {
            "question": prompt_text[-200:],
            "generated_sql": gen_sql,
            "ground_truth": gt_sql,
            "exact_match": exact_match,
            "exec_match": exec_match,
            "error": error_type,
        }
        results.append(entry)

        eval_fh.write(f"--- Example {i+1} ---\n")
        eval_fh.write(f"Question: {prompt_text[-200:]}\n")
        eval_fh.write(f"Generated: {gen_sql}\n")
        eval_fh.write(f"Ground truth: {gt_sql}\n")
        eval_fh.write(f"Exact match: {exact_match}, Exec match: {exec_match}, Error: {error_type}\n\n")

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{len(examples)} | exec_acc={execution_matches/(i+1):.3f}")

    eval_fh.close()

    n = len(results)
    report = {
        "execution_accuracy": execution_matches / n if n else 0,
        "exact_match_accuracy": exact_matches / n if n else 0,
        "total": n,
        "errors": error_counts,
        "per_example": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEval complete: exec_acc={report['execution_accuracy']:.3f}, exact_match={report['exact_match_accuracy']:.3f}")
    print(f"Errors: {error_counts}")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add projects/nlp/text2sql_finetune/evaluate.py
git commit -m "feat: add evaluate.py — SQL generation + execution-match with timeout protection"
```

---

### Task 7: test_evaluate.py — SQL execution, extraction, timeout

**Files:**
- Create: `tests/test_text2sql_evaluate.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for evaluate.py — SQL extraction, execution, timeout."""
import time
from projects.nlp.text2sql_finetune.evaluate import (
    extract_sql,
    parse_create_table,
    execute_sql,
    execute_with_timeout,
)

ASSISTANT_OUTPUT = """<|im_start|>system
You are a SQL expert.
<|im_end|>
<|im_start|>user
Schema:
CREATE TABLE stadium (id int, name text, capacity int);

Question: total capacity?
<|im_end|>
<|im_start|>assistant
SELECT SUM(capacity) FROM stadium
<|im_end|>"""


def test_extract_sql_from_assistant_tags():
    sql = extract_sql(ASSISTANT_OUTPUT)
    assert sql == "SELECT SUM(capacity) FROM stadium", f"Got: {sql}"


def test_extract_sql_multiple_tags():
    text = "<|im_start|>assistant\nSELECT 1<|im_end|>\n<|im_start|>assistant\nSELECT 2<|im_end|>"
    sql = extract_sql(text)
    assert sql == "SELECT 2", f"Got: {sql}"  # Last assistant block


def test_parse_create_table_single():
    context = "CREATE TABLE t1 (a int, b text);"
    tables = parse_create_table(context)
    assert len(tables) == 1
    assert "CREATE TABLE t1" in tables[0]


def test_parse_create_table_multiple():
    context = "CREATE TABLE t1 (a int);\nSome text\nCREATE TABLE t2 (x real);"
    tables = parse_create_table(context)
    assert len(tables) == 2


def test_parse_create_table_case_insensitive():
    context = "create table t1 (a int);"
    tables = parse_create_table(context)
    assert len(tables) == 1


def test_execute_sql_correct():
    create_tables = ["CREATE TABLE t (x int, y text)", "INSERT INTO t VALUES (1, 'a'), (2, 'b')"]
    success, result = execute_sql(create_tables, "SELECT x FROM t WHERE y = 'a'")
    assert success
    assert result == [(1,)], f"Got: {result}"


def test_execute_sql_syntax_error():
    success, result = execute_sql(["CREATE TABLE t (x int)"], "SELEC x FROM t")
    assert not success
    assert "syntax" in result.lower() or "error" in result.lower() or "near" in result.lower()


def test_execute_sql_table_not_found():
    success, result = execute_sql(["CREATE TABLE t (x int)"], "SELECT x FROM nonexistent")
    assert not success


def test_execute_sql_timeout():
    """A slow Cartesian product should time out."""
    create_tables = ["CREATE TABLE a (x int)", "CREATE TABLE b (y int)"]
    # Insert enough rows to make a cross join slow
    for i in range(100):
        create_tables.append(f"INSERT INTO a VALUES ({i})")
        create_tables.append(f"INSERT INTO b VALUES ({i})")

    success, result = execute_with_timeout(
        create_tables,
        "SELECT * FROM a CROSS JOIN b CROSS JOIN a c CROSS JOIN b d CROSS JOIN a e",
        timeout=1,
    )
    assert not success
    assert result == "__TIMEOUT__", f"Got: {result}"
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_text2sql_evaluate.py -v
```

Expected: 9 PASS (note: timeout test may take ~1s).

- [ ] **Step 3: Commit**

```bash
git add tests/test_text2sql_evaluate.py
git commit -m "test: add SQL extraction, execution, and timeout tests"
```

---

### Task 8: launch.py — Colab bootstrap

**Files:**
- Create: `projects/nlp/text2sql_finetune/launch.py`

- [ ] **Step 1: Write launch.py**

```python
"""Colab bootstrap for text2sql_finetune.

Upload all source files, then run:
    colab exec -s <name> -f launch.py

Outputs go to /content/text2sql_finetune-output/
"""
import os
import subprocess
import sys
import shutil

PROJECT_DIR = "/content/text2sql_finetune"
OUTPUT_DIR = "/content/text2sql_finetune-output"
DEPS = ["peft", "datasets", "accelerate", "torch"]

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Clear stale HF cache ---
hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    print("[launch] Cleared HF datasets cache")

# --- Set env ---
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Prepare dataset ---
print("[launch] Preparing dataset...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "train", "--max_examples", "500",
    "--output", f"{PROJECT_DIR}/data/train.pt",
])
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "test", "--max_examples", "100",
    "--output", f"{PROJECT_DIR}/data/test.pt",
])
print("[launch] Dataset prepared")

# --- Train ---
print("[launch] Starting training...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/train.py",
    "--data_path", f"{PROJECT_DIR}/data/train.pt",
    "--output_dir", f"{PROJECT_DIR}/lora_weights",
    "--max_steps", "0",
])
print("[launch] Training complete")

# --- Evaluate ---
print("[launch] Running evaluation...")
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/evaluate.py",
    "--data_path", f"{PROJECT_DIR}/data/test.pt",
    "--lora_path", f"{PROJECT_DIR}/lora_weights",
    "--output", f"{OUTPUT_DIR}/eval_report.json",
])
print("[launch] Evaluation complete")

# --- Gather outputs ---
for src in [f"{PROJECT_DIR}/logs/train.log", f"{PROJECT_DIR}/logs/metrics.csv",
            f"{PROJECT_DIR}/logs/eval.log", f"{OUTPUT_DIR}/eval_report.json"]:
    if os.path.exists(src):
        dst = os.path.join(OUTPUT_DIR, os.path.relpath(src, PROJECT_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

# --- Tar outputs ---
subprocess.check_call([
    "tar", "-czf", "/content/text2sql_finetune-output.tar.gz",
    "-C", "/content", "text2sql_finetune-output",
])
print("[launch] DONE. Outputs at /content/text2sql_finetune-output.tar.gz")
print("[launch] Download: colab download -s <session> /content/text2sql_finetune-output.tar.gz")
```

- [ ] **Step 2: Commit**

```bash
git add projects/nlp/text2sql_finetune/launch.py
git commit -m "feat: add launch.py — Colab bootstrap for text2sql fine-tuning"
```

---

### Task 9: fetch.sh + tar_outputs.py — cron monitoring

**Files:**
- Create: `projects/nlp/text2sql_finetune/tar_outputs.py`
- Create: `projects/nlp/text2sql_finetune/fetch.sh`

- [ ] **Step 1: Write tar_outputs.py**

```python
"""Uploaded to Colab VM. tar_outputs.py creates a tarball of the output directory."""
import subprocess
import sys

OUTPUT_DIR = "/content/text2sql_finetune-output"
TAR_PATH = "/content/text2sql_finetune-output.tar.gz"

result = subprocess.run(
    ["tar", "-czf", TAR_PATH, "-C", "/content", "text2sql_finetune-output"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"tar failed: {result.stderr}", file=sys.stderr)
    sys.exit(1)
print(f"Created {TAR_PATH}")
```

- [ ] **Step 2: Write fetch.sh**

```bash
#!/bin/bash
# Cron watchtower payload for text2sql_finetune.
# Fires every 2 min from CronCreate. Stops when eval_report.json appears.
#
# Usage: SESSION=<name> bash fetch.sh
set -euo pipefail

SESSION="${SESSION:?must set SESSION env var}"
OUTPUT_DIR="/tmp/text2sql-output-$$"
mkdir -p "$OUTPUT_DIR"

# 1. Check session alive
echo "=== Checking session $SESSION ==="
if ! colab sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "FATAL: session $SESSION not found — stopping cron"
    exit 1
fi

# 2. Tar outputs on VM
echo "=== Tarring outputs on VM ==="
colab exec -s "$SESSION" -f tar_outputs.py --timeout 15 || {
    echo "WARNING: tar failed, trying individual file download"
    colab download -s "$SESSION" /content/text2sql_finetune-output/eval_report.json "$OUTPUT_DIR/eval_report.json" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql_finetune-output/logs/train.log "$OUTPUT_DIR/train.log" 2>/dev/null || true
    colab download -s "$SESSION" /content/text2sql_finetune-output/logs/metrics.csv "$OUTPUT_DIR/metrics.csv" 2>/dev/null || true
}

# 3. Download tarball (if tar succeeded)
if [ -f "output.tar.gz" ] || colab download -s "$SESSION" /content/text2sql_finetune-output.tar.gz "$OUTPUT_DIR/output.tar.gz" 2>/dev/null; then
    if [ -f "output.tar.gz" ]; then
        mv output.tar.gz "$OUTPUT_DIR/output.tar.gz"
    fi
    tar -xzf "$OUTPUT_DIR/output.tar.gz" -C "$OUTPUT_DIR" 2>/dev/null || true
fi

# 4. Report
echo ""
echo "=== train.log (last 5 lines) ==="
tail -5 "$OUTPUT_DIR/logs/train.log" 2>/dev/null || echo "(no train.log yet)"

echo ""
echo "=== metrics.csv (last 3 lines) ==="
tail -3 "$OUTPUT_DIR/metrics.csv" 2>/dev/null || echo "(no metrics.csv yet)"

echo ""
echo "=== Eval report ==="
if [ -f "$OUTPUT_DIR/eval_report.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/eval_report.json') as f:
    r = json.load(f)
print(f\"exec_acc={r['execution_accuracy']:.3f}  exact_match={r['exact_match_accuracy']:.3f}  total={r['total']}\")
print(f\"errors: {r['errors']}\")
"
    echo "DONE — eval complete. Remove cron job."
else
    echo "(no eval report yet — training in progress)"
fi

rm -rf "$OUTPUT_DIR"
```

- [ ] **Step 3: Make fetch.sh executable**

```bash
chmod +x projects/nlp/text2sql_finetune/fetch.sh
```

- [ ] **Step 4: Commit**

```bash
git add projects/nlp/text2sql_finetune/tar_outputs.py projects/nlp/text2sql_finetune/fetch.sh
git commit -m "feat: add fetch.sh and tar_outputs.py for cron watchtower monitoring"
```

---

### Task 10: Integration test — end-to-end with 3 examples

**Files:**
- Create: `tests/test_integration_text2sql.py`

- [ ] **Step 1: Write integration test**

```python
"""End-to-end test: dataset → train → eval on 3 hand-crafted examples."""
import json
import os
import subprocess
import sys
import tempfile

PROJECT_DIR = "projects/nlp/text2sql_finetune"


def test_e2e_minimal():
    """Run full pipeline with 3 examples. Verifies all modules connect correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_data.pt")
        lora_path = os.path.join(tmpdir, "lora_weights")
        eval_path = os.path.join(tmpdir, "eval_report.json")

        # 1. Create dataset (using real HF dataset, 3 examples)
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/dataset.py",
            "--split", "train", "--max_examples", "3",
            "--output", data_path,
        ], check=True)

        assert os.path.exists(data_path), "dataset not saved"

        # 2. Train (3 examples, 2 steps)
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/train.py",
            "--data_path", data_path,
            "--output_dir", lora_path,
            "--max_steps", "2",
            "--batch_size", "2",
            "--grad_accum", "1",
        ], check=True)

        assert os.path.exists(os.path.join(lora_path, "adapter_config.json")), "adapter not saved"

        # 3. Eval
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/evaluate.py",
            "--data_path", data_path,
            "--lora_path", lora_path,
            "--output", eval_path,
        ], check=True)

        assert os.path.exists(eval_path), "eval report not saved"
        with open(eval_path) as f:
            report = json.load(f)
        assert "execution_accuracy" in report
        assert report["total"] == 3
        print(f"E2E passed: exec_acc={report['execution_accuracy']:.3f}")
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest tests/test_integration_text2sql.py::test_e2e_minimal -v -s
```

Expected: PASS (downloads HF dataset + Qwen3-0.6B model ~2 min first time, then fast).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_text2sql.py
git commit -m "test: add end-to-end integration test (dataset → train → eval)"
```

---

### Task 11: Deploy to Colab

- [ ] **Step 1: Warmup session (cache model + dataset)**

```bash
colab new --gpu T4 -s text2sql-warmup
colab upload -s text2sql-warmup projects/nlp/text2sql_finetune/ --remote /content/text2sql_finetune/
colab exec -s text2sql-warmup -c "python /content/text2sql_finetune/dataset.py --split train --max_examples 5 --output /tmp/test.pt"
colab stop -s text2sql-warmup
```

- [ ] **Step 2: Real training session**

```bash
colab new --gpu T4 -s text2sql-train
colab upload -s text2sql-train projects/nlp/text2sql_finetune/ --remote /content/text2sql_finetune/
colab exec -s text2sql-train -f launch.py --timeout 540
```

- [ ] **Step 3: Set up cron monitoring (from another terminal or via CronCreate)**

```
CronCreate cron="*/2 * * * *" prompt="SESSION=text2sql-train bash projects/nlp/text2sql_finetune/fetch.sh" recurring=true
```

- [ ] **Step 4: After training completes, download outputs**

```bash
colab download -s text2sql-train /content/text2sql_finetune-output.tar.gz
tar -xzf text2sql_finetune-output.tar.gz -C projects/nlp/text2sql_finetune/output/
```

- [ ] **Step 5: Review eval results**

```bash
cat projects/nlp/text2sql_finetune/output/eval_report.json | python -m json.tool | head -20
```

---
