"""Evaluate fine-tuned model via execution-match against in-memory SQLite databases.

Usage:
    python evaluate.py --data_path data/test.pt --lora_path lora_weights/ --output eval_report.json
"""
import argparse
import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = "You are a SQL expert. Given a table schema and a question, write the correct SQL query."
SQL_TIMEOUT = 5
LOG_DIR = "logs"


def extract_sql(text):
    """Extract SQL from model output, stripping Qwen3 think tags."""
    pattern = r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    raw = matches[-1].strip() if matches else ""
    if not raw:
        idx = text.rfind("assistant\n")
        raw = text[idx + len("assistant\n"):].strip() if idx != -1 else text.strip()
    # Strip Qwen3 thinking tags
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    return raw


def parse_create_table(sql):
    """Extract CREATE TABLE statements from context string.

    Dataset format: "CREATE TABLE name (cols)" with optional trailing ";".
    Use parenthesis-bounded match — avoids over-matching into question text.
    """
    return re.findall(r"CREATE\s+TABLE\s+\w+\s*\([^)]+\)", sql, re.IGNORECASE)


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
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        # Find where labels start (first non -100 position)
        label_start = 0
        for j in range(len(labels)):
            if labels[j] != -100:
                label_start = j
                break

        prompt_ids = input_ids[:label_start]
        gt_ids = [int(input_ids[j]) for j in range(label_start, len(labels)) if labels[j] != -100]

        gt_sql = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

        # Use raw prompt token IDs directly (preserves chat template structure)
        prompt_tensor = prompt_ids.unsqueeze(0).to(device)
        attn_tensor = ex["attention_mask"][:label_start].unsqueeze(0).to(device)
        model_inputs = {"input_ids": prompt_tensor, "attention_mask": attn_tensor}
        with torch.no_grad():
            generated = model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
        gen_sql = extract_sql(full_output)

        # Parse CREATE TABLE from the context text
        context_text = tokenizer.decode(input_ids[:label_start].tolist(), skip_special_tokens=True)
        create_tables = parse_create_table(context_text)
        if not create_tables:
            create_tables = parse_create_table(tokenizer.decode(input_ids.tolist(), skip_special_tokens=True))

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
            "question": context_text[-200:],
            "generated_sql": gen_sql,
            "ground_truth": gt_sql,
            "exact_match": exact_match,
            "exec_match": exec_match,
            "error": error_type,
        }
        results.append(entry)

        eval_fh.write(f"--- Example {i+1} ---\n")
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
