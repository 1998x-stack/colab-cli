"""Load s1K from HuggingFace, filter by quality/difficulty/diversity, save JSONL.

Three-stage filtering pipeline:
  1. Quality: remove samples missing required content
  2. Difficulty: keep only questions base model gets wrong (or trace_len > median with --skip-difficulty)
  3. Diversity: sample evenly across trace length deciles

Usage:
    python dataset.py                          # uses default HF token path
    HF_TOKEN=xxx python dataset.py             # explicit token
    python dataset.py --output s1k_filtered.jsonl
    python dataset.py --skip-difficulty        # skip base model eval (faster)
    python dataset.py --n-total 500 --seed 0
"""
import argparse, json, os, random, re, sys
from pathlib import Path

TOKEN_PATHS = [
    os.path.expanduser("~/.huggingface/access_token"),
    os.path.join(os.path.dirname(__file__), "..", "..", ".huggingface", "access_token"),
]

# Markers used in the s1K formatting (added during format_sample, not in raw data)
THINK_MARKER = "<|im_start|>think"
ANSWER_MARKER = "<|im_start|>answer"


def get_hf_token():
    """Read HF token from standard locations or env var."""
    for p in TOKEN_PATHS:
        try:
            with open(p) as f:
                return f.read().strip()
        except FileNotFoundError:
            continue
    return os.environ.get("HF_TOKEN", None)


def load_s1k(hf_token):
    """Load s1K from HF. Returns list of dicts with keys: question, trace, solution.

    Raw data fields (verified 2026-06-11):
      - question (str): the math problem
      - thinking_trajectories (list[str], always 1 element): reasoning trace
      - solution (str): final solution/answer
      - cot_type, source_type, metadata, cot, attempt: additional fields, unused here
    """
    from datasets import load_dataset
    ds = load_dataset("simplescaling/s1K", token=hf_token)
    # Dataset has a single split — use the first (and only) split
    split = list(ds.keys())[0]
    items = []
    for row in ds[split]:
        # thinking_trajectories is always a list with 1 element in s1K
        trace = row["thinking_trajectories"][0] if row["thinking_trajectories"] else ""
        items.append({
            "question": row["question"],
            "trace": trace,
            "solution": row["solution"],
        })
    return items


def filter_quality(items):
    """Remove samples with empty question, trace, or solution.

    Note: s1K raw data does NOT contain <|im_start|>think / <|im_start|>answer
    markers — those are inserted by format_sample(). The quality check here
    simply validates that all content fields are present and non-empty.
    """
    kept = []
    dropped = 0
    for item in items:
        q = (item.get("question") or "").strip()
        trace = (item.get("trace") or "").strip()
        sol = (item.get("solution") or "").strip()

        if not q or not trace or not sol:
            dropped += 1
            continue

        kept.append(item)
    print(f"[quality] kept {len(kept)}, dropped {dropped}")
    return kept


def filter_difficulty(items, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
    """Remove samples the base model already gets right. Keeps only hard questions.

    Uses batch generation with temperature=0 for deterministic eval.
    Returns items base model got WRONG (hard ones), annotated with trace_len and base_correct.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[difficulty] loading {model_name} for zero-shot eval...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hard_items = []
    batch_size = 4
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = []
        for item in batch:
            prompt = (
                f"<|im_start|>user\n{item['question']}\n\n"
                f"Provide your final answer within \\boxed{{}}."
                f"<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, output_ids in enumerate(outputs):
            response = tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            correct = check_correctness(batch[j]["solution"], response)
            if not correct:
                trace_len = len(tokenizer.encode(batch[j].get("trace", "") or ""))
                hard_items.append({**batch[j], "trace_len": trace_len, "base_correct": False})
            # Drop correct items — only keep hard ones

        if (i // batch_size) % 10 == 0:
            print(f"  [difficulty] {i}/{len(items)} done, {len(hard_items)} hard so far")

    print(f"[difficulty] kept {len(hard_items)} hard (base model got wrong)")
    return hard_items


def check_correctness(reference_solution, model_answer):
    """Check if model answer matches reference. Extracts \\boxed{...} and normalizes."""
    # Try to extract \\boxed{...} from model answer
    boxed = re.findall(r'\\boxed\{([^}]+)\}', model_answer)
    if boxed:
        model_final = boxed[-1].strip()
    else:
        # Fallback: last non-empty line
        lines = [l.strip() for l in model_answer.strip().split('\n') if l.strip()]
        model_final = lines[-1] if lines else model_answer.strip()

    ref_boxed = re.findall(r'\\boxed\{([^}]+)\}', reference_solution)
    if ref_boxed:
        ref_final = ref_boxed[-1].strip()
    else:
        ref_final = reference_solution.strip()

    # Normalize and compare
    def normalize(s):
        s = s.replace(' ', '').replace(',', '').lower()
        s = s.rstrip('.;:!?')
        return s

    return normalize(model_final) == normalize(ref_final)


def filter_diversity(items, n_total=300):
    """Sample evenly across trace length deciles as proxy for domain diversity.

    Sorts by trace_len, splits into 10 deciles, samples proportionally from each.
    Trims to exact n_total via random shuffle.
    """
    items_sorted = sorted(items, key=lambda x: x.get("trace_len", 0))
    selected = []
    for d in range(10):
        start = d * len(items_sorted) // 10
        end = (d + 1) * len(items_sorted) // 10
        decile = items_sorted[start:end]
        if decile:
            n_per_decile = max(1, n_total // 10)
            n = min(n_per_decile, len(decile))
            selected.extend(random.sample(decile, n))
    # Trim to exact n_total
    random.shuffle(selected)
    selected = selected[:n_total]
    print(f"[diversity] selected {len(selected)} from {len(items)}")
    return selected


def format_sample(item):
    """Format a sample as the full training string with think/answer delimiters."""
    q = item["question"].strip()
    trace = (item.get("trace") or "").strip()
    sol = (item.get("solution") or "").strip()

    return (
        f"<|im_start|>user\n{q}\n<|im_start|>assistant\n"
        f"{THINK_MARKER}\n{trace}\n{ANSWER_MARKER}\n{sol}<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load s1K from HF, apply quality/difficulty/diversity filters, save JSONL."
    )
    parser.add_argument("--output", default="s1k_filtered.jsonl",
                       help="Output JSONL path (default: s1k_filtered.jsonl)")
    parser.add_argument("--n-total", type=int, default=300,
                       help="Target sample count after filtering (default: 300)")
    parser.add_argument("--skip-difficulty", action="store_true",
                       help="Skip base model eval; use trace_len > median as difficulty proxy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: No HF token found. Set HF_TOKEN env var or create .huggingface/access_token")
        sys.exit(1)
    print(f"[dataset] HF token found ({hf_token[:8]}...)")

    # 1. Load
    print("[dataset] Loading s1K from HuggingFace...")
    items = load_s1k(hf_token)
    print(f"[dataset] Loaded {len(items)} raw samples")

    # 2. Quality filter
    items = filter_quality(items)

    # 3. Difficulty filter
    if args.skip_difficulty:
        # Use trace length as rough difficulty proxy (no GPU needed)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
        )
        for item in items:
            item["trace_len"] = len(tokenizer.encode(
                item.get("trace", "") or ""
            ))
        items_sorted = sorted(items, key=lambda x: x["trace_len"])
        median_len = items_sorted[len(items) // 2]["trace_len"]
        items = [item for item in items if item["trace_len"] > median_len]
        print(f"[difficulty-skip] kept {len(items)} with trace_len > median ({median_len})")
    else:
        items = filter_difficulty(items)

    if len(items) < args.n_total:
        print(f"[dataset] WARNING: only {len(items)} items after difficulty filter, need {args.n_total}")
        args.n_total = len(items)

    # 4. Diversity filter
    items = filter_diversity(items, n_total=args.n_total)

    # 5. Save filtered JSONL
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            formatted = format_sample(item)
            f.write(json.dumps({
                "text": formatted,
                "question": item["question"],
                "solution": item.get("solution", ""),
            }) + "\n")
    print(f"[dataset] Saved {len(items)} formatted samples to {output_path}")

    # 6. Save metadata
    meta_path = output_path.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_total": len(items),
            "avg_trace_len": round(
                sum(it.get("trace_len", 0) for it in items) / max(len(items), 1), 1
            ),
            "seed": args.seed,
        }, f, indent=2)
    print(f"[dataset] Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
