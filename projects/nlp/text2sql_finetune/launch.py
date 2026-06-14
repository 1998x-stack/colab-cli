"""Colab bootstrap for text2sql_finetune — single-shot inline execution.

NOTE: Prefer bg_launch.py for new deployments. This script inlines training/eval
code to avoid torchao subprocess import issues, but duplicates logic from train.py
and evaluate.py. bg_launch.py spawns train.py as a detached subprocess — survives
WebSocket drops and keeps a single source of truth.

Use this only if bg_launch.py fails with torchao import errors in subprocess.

Usage:
    colab exec -s <name> -f launch.py --timeout 540

Outputs go to /content/text2sql-finetune-output/
"""
import os
import subprocess
import sys
import shutil

PROJECT_DIR = "/content/text2sql_finetune"
OUTPUT_DIR = "/content/text2sql-finetune-output"
DEPS = ["peft", "datasets", "accelerate", "torchao"]

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + DEPS,
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

# --- Prepare dataset (only train split exists) ---
print("[launch] Preparing dataset...")
import torch as _torch
subprocess.check_call([
    sys.executable, f"{PROJECT_DIR}/dataset.py",
    "--split", "train", "--max_examples", "600",
    "--output", f"{PROJECT_DIR}/data/all.pt",
])
all_data = _torch.load(f"{PROJECT_DIR}/data/all.pt", weights_only=False)
_torch.save(all_data[:500], f"{PROJECT_DIR}/data/train.pt")
_torch.save(all_data[500:600], f"{PROJECT_DIR}/data/test.pt")
print(f"[launch] Dataset prepared: 500 train, 100 test")

# --- Train (inline, no subprocess — avoids torchao version issues) ---
print("[launch] Starting training...")
import time
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-0.6B"

class _TensorDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def _collate_fn(batch):
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids = _torch.full((len(batch), max_len), 0, dtype=_torch.long)
    attention_mask = _torch.full((len(batch), max_len), 0, dtype=_torch.long)
    labels = _torch.full((len(batch), max_len), -100, dtype=_torch.long)
    for i, ex in enumerate(batch):
        n = ex["input_ids"].shape[0]
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        labels[i, :n] = ex["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

device = _torch.device("cuda")
examples = _torch.load(f"{PROJECT_DIR}/data/train.pt", weights_only=False)
dataset = _TensorDataset(examples)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=_collate_fn, drop_last=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=_torch.bfloat16, device_map="auto", trust_remote_code=True,
)
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.train()
print(f"[launch] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = _torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = _torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

LOG_DIR = f"{PROJECT_DIR}/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_fh = open(f"{LOG_DIR}/train.log", "w")
metrics_path = f"{LOG_DIR}/metrics.csv"
with open(metrics_path, "w", newline="") as f:
    csv.writer(f).writerow(["step", "loss", "lr", "elapsed_s"])

step = 0
t0 = time.time()
GRAD_ACCUM = 2

for batch_idx, batch in enumerate(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss / GRAD_ACCUM
    loss.backward()

    if (batch_idx + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        train_loss = loss.item() * GRAD_ACCUM
        line = f"[{time.strftime('%H:%M:%S')}] step {step} | loss={train_loss:.4f} | lr={lr:.2e} | elapsed={elapsed:.0f}s"
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([step, train_loss, lr, elapsed])

total_elapsed = time.time() - t0
line = f"[{time.strftime('%H:%M:%S')}] Training complete | steps={step} | total_time={total_elapsed:.0f}s"
print(line)
log_fh.write(line + "\n")
log_fh.close()

os.makedirs(f"{PROJECT_DIR}/lora_weights", exist_ok=True)
model.save_pretrained(f"{PROJECT_DIR}/lora_weights")
print("[launch] Training complete — LoRA weights saved")

# --- Evaluate (inline) ---
print("[launch] Running evaluation...")
import json, re, sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def _extract_sql(text):
    pattern = r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    raw = matches[-1].strip() if matches else ""
    if not raw:
        idx = text.rfind("assistant\n")
        raw = text[idx + len("assistant\n"):].strip() if idx != -1 else text.strip()
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    return raw

def _parse_create_table(sql):
    return re.findall(r"CREATE\s+TABLE\s+\w+\s*\([^)]+\)", sql, re.IGNORECASE)

def _execute_sql(create_tables, query):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        for ct in create_tables:
            conn.execute(ct)
        conn.commit()
        cursor = conn.execute(query)
        return True, [tuple(row) for row in cursor.fetchall()]
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def _execute_with_timeout(create_tables, query, timeout=5):
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            return executor.submit(_execute_sql, create_tables, query).result(timeout=timeout)
        except FuturesTimeoutError:
            return False, "__TIMEOUT__"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=_torch.bfloat16, device_map="auto", trust_remote_code=True,
)
eval_model = PeftModel.from_pretrained(base_model, f"{PROJECT_DIR}/lora_weights")
eval_model.eval()

test_examples = _torch.load(f"{PROJECT_DIR}/data/test.pt", weights_only=False)
results = []
error_counts = {"syntax": 0, "timeout": 0, "wrong_result": 0, "other": 0}
exec_matches = 0
exact_matches = 0

eval_fh = open(f"{LOG_DIR}/eval.log", "w")

for i, ex in enumerate(test_examples):
    input_ids = ex["input_ids"]
    labels = ex["labels"]
    label_start = next((j for j in range(len(labels)) if labels[j] != -100), len(labels))
    prompt_ids = input_ids[:label_start]
    gt_ids = [int(input_ids[j]) for j in range(label_start, len(labels)) if labels[j] != -100]

    prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    gt_sql = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

    # Use raw token IDs for generation (preserves chat template)
    prompt_tensor = input_ids[:label_start].unsqueeze(0).to(device)
    attn_tensor = ex["attention_mask"][:label_start].unsqueeze(0).to(device)

    with _torch.no_grad():
        generated = eval_model.generate(
            input_ids=prompt_tensor, attention_mask=attn_tensor,
            max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
    gen_sql = _extract_sql(full_output)

    context_text = tokenizer.decode(input_ids[:label_start].tolist(), skip_special_tokens=True)
    create_tables = _parse_create_table(context_text)
    if not create_tables:
        create_tables = _parse_create_table(tokenizer.decode(input_ids.tolist(), skip_special_tokens=True))

    exact_match = (gen_sql.strip().lower() == gt_sql.strip().lower())
    exec_match = False
    error_type = None

    if gen_sql:
        success, result = _execute_with_timeout(create_tables, gen_sql)
        if success:
            gt_success, gt_result = _execute_sql(create_tables, gt_sql)
            if gt_success:
                exec_match = (result == gt_result)
                if not exec_match:
                    error_type = "wrong_result"
            else:
                error_type = "other"
        else:
            error_type = "timeout" if result == "__TIMEOUT__" else "syntax"
    else:
        error_type = "syntax"

    if exec_match:
        exec_matches += 1
    if exact_match:
        exact_matches += 1
    if error_type:
        error_counts[error_type] += 1

    results.append({
        "question": context_text[-200:],
        "generated_sql": gen_sql,
        "ground_truth": gt_sql,
        "exact_match": exact_match,
        "exec_match": exec_match,
        "error": error_type,
    })

    eval_fh.write(f"--- Example {i+1} ---\nGenerated: {gen_sql}\nGround truth: {gt_sql}\n")
    eval_fh.write(f"Exact match: {exact_match}, Exec match: {exec_match}, Error: {error_type}\n\n")

    if (i + 1) % 10 == 0:
        print(f"  Evaluated {i+1}/{len(test_examples)} | exec_acc={exec_matches/(i+1):.3f}")

eval_fh.close()

n = len(results)
report = {
    "execution_accuracy": exec_matches / n if n else 0,
    "exact_match_accuracy": exact_matches / n if n else 0,
    "total": n,
    "errors": error_counts,
    "per_example": results,
}
with open(f"{OUTPUT_DIR}/eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[launch] Eval: exec_acc={report['execution_accuracy']:.3f}, exact_match={report['exact_match_accuracy']:.3f}")
print(f"[launch] Errors: {error_counts}")

# --- Gather outputs ---
for src in [f"{LOG_DIR}/train.log", f"{LOG_DIR}/metrics.csv",
            f"{LOG_DIR}/eval.log", f"{OUTPUT_DIR}/eval_report.json"]:
    if os.path.exists(src):
        dst = os.path.join(OUTPUT_DIR, os.path.relpath(src, PROJECT_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

# --- Tar outputs ---
subprocess.check_call([
    "tar", "-czf", "/content/text2sql-finetune-output.tar.gz",
    "-C", "/content", "text2sql-finetune-output",
])
print("[launch] DONE. Outputs at /content/text2sql-finetune-output.tar.gz")
print(f"[launch] Download: colab download -s <session> /content/text2sql-finetune-output.tar.gz")
