"""Check training progress, run eval if done, tar outputs for download.
Run as: colab exec -s <name> -f check_and_eval.py --timeout 120
"""
import os, sys, subprocess, shutil, json, re, sqlite3, time, csv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

PROJECT_DIR = "/content/text2sql_finetune"
OUTPUT_DIR = "/content/text2sql-finetune-output"

# Check training status
log_path = f"{PROJECT_DIR}/logs/train.log"
lora_path = f"{PROJECT_DIR}/lora_weights"
adapter_path = f"{lora_path}/adapter_config.json"

if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    print("=== train.log (last 5) ===")
    for line in lines[-5:]:
        print(line.rstrip())
else:
    print("No train.log yet — training may not have started")

if os.path.exists(adapter_path):
    print("\n=== Training complete! Running eval ===")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def extract_sql(text):
        pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        idx = text.rfind("assistant\n")
        if idx != -1:
            return text[idx + len("assistant\n"):].strip()
        return text.strip()

    def parse_create_table(sql):
        return re.findall(r"CREATE\s+TABLE\s+[^;]+;", sql, re.IGNORECASE)

    def execute_sql(create_tables, query):
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

    def execute_with_timeout(create_tables, query, timeout=5):
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return executor.submit(execute_sql, create_tables, query).result(timeout=timeout)
            except FuturesTimeoutError:
                return False, "__TIMEOUT__"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    eval_model = PeftModel.from_pretrained(base_model, lora_path)
    eval_model.eval()

    test_examples = torch.load(f"{PROJECT_DIR}/data/test.pt", weights_only=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)

    results = []
    error_counts = {"syntax": 0, "timeout": 0, "wrong_result": 0, "other": 0}
    exec_matches = 0
    exact_matches = 0

    eval_fh = open(f"{PROJECT_DIR}/logs/eval.log", "w")

    for i, ex in enumerate(test_examples):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        label_start = next((j for j in range(len(labels)) if labels[j] != -100), len(labels))
        prompt_ids = input_ids[:label_start]
        gt_ids = [int(input_ids[j]) for j in range(label_start, len(labels)) if labels[j] != -100]

        gt_sql = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

        prompt_tensor = input_ids[:label_start].unsqueeze(0).to(device)
        attn_tensor = ex["attention_mask"][:label_start].unsqueeze(0).to(device)

        with torch.no_grad():
            generated = eval_model.generate(
                input_ids=prompt_tensor, attention_mask=attn_tensor,
                max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
        gen_sql = extract_sql(full_output)

        context_text = tokenizer.decode(input_ids[:label_start].tolist(), skip_special_tokens=True)
        create_tables = parse_create_table(context_text)
        if not create_tables:
            create_tables = parse_create_table(tokenizer.decode(input_ids.tolist(), skip_special_tokens=True))

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
            "generated_sql": gen_sql,
            "ground_truth": gt_sql,
            "exact_match": exact_match,
            "exec_match": exec_match,
            "error": error_type,
        })
        eval_fh.write(f"--- {i+1} ---\nGen: {gen_sql}\nGT: {gt_sql}\nMatch: {exec_match} Err: {error_type}\n\n")

        if (i + 1) % 20 == 0:
            print(f"  Eval {i+1}/{len(test_examples)} | exec_acc={exec_matches/(i+1):.3f}")

    eval_fh.close()

    n = len(results)
    report = {
        "execution_accuracy": exec_matches / n if n else 0,
        "exact_match_accuracy": exact_matches / n if n else 0,
        "total": n,
        "errors": error_counts,
    }
    with open(f"{OUTPUT_DIR}/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEval complete: exec_acc={report['execution_accuracy']:.3f}, exact_match={report['exact_match_accuracy']:.3f}")
    print(f"Errors: {error_counts}")

    # Gather outputs
    for src in [f"{PROJECT_DIR}/logs/train.log", f"{PROJECT_DIR}/logs/metrics.csv",
                f"{PROJECT_DIR}/logs/eval.log", f"{OUTPUT_DIR}/eval_report.json"]:
        if os.path.exists(src):
            dst = os.path.join(OUTPUT_DIR, os.path.relpath(src, PROJECT_DIR))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # Tar
    subprocess.check_call([
        "tar", "-czf", "/content/text2sql-finetune-output.tar.gz",
        "-C", "/content", "text2sql-finetune-output",
    ])
    print("DONE. Tarball at /content/text2sql-finetune-output.tar.gz")
else:
    print("\nTraining still in progress — check back later")
    # Check PID
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "train.py" in line and "grep" not in line:
            print(f"  Process: {line.strip()}")
