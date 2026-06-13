#!/usr/bin/env python3
"""Multi-config evaluation with s1 metrics.

Evaluates trained Qwen2.5-7B-Instruct + LoRA adapter on MATH500 subset
with multiple Budget Forcing configurations, computing the s1 paper's
three core metrics: Control, Scaling, and Performance.

Usage:
    python evaluate.py --adapter /content/s1-t4/checkpoints/adapter_final --eval_data eval_data.jsonl
    python evaluate.py --adapter /path/to/adapter --eval_data data.jsonl --skip_baselines
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from tqdm import tqdm


# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOG_DIR = "/content/s1-t4/logs"

# BF Configs: (name, max_thinking_tokens, num_suppressions, max_new_tokens)
BF_CONFIGS = [
    ("base_no_bf", 0, 0, 4096),      # No BF
    ("bf_2048_1x", 2048, 1, 4096),   # BF 2048 max + 1 suppression
    ("bf_4096_2x", 4096, 2, 8192),   # BF 4096 max + 2 suppressions
]

BASELINE_CONFIGS = [
    ("base_cot", 0, 0, 4096),        # Base model w/o adapter
    ("base_cot_bf", 2048, 1, 4096),  # Base model + BF
]

EVAL_PROMPT_TEMPLATE = (
    "<|im_start|>user\n{question}\n\n"
    "Think step by step and provide your final answer within \\boxed{{}}.\n"
    "<|im_start|>assistant\n"
)

# Handle max_thinking_tokens==0 (no budget): pass this large value instead
# so BudgetForcingLogitsProcessor never triggers budget-limited forcing.
NO_BUDGET_SENTINEL = 999999


# --- Logging ---

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure structured logging to file and stdout with timestamps."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")

    logger = logging.getLogger("s1-t4-eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(sh)

    return logger


# --- Answer extraction ---

def extract_boxed_answer(text: str) -> str | None:
    """Extract content inside the first \\boxed{...}, handling nested braces.

    Uses brace-depth counting to correctly handle nested curly braces
    inside the boxed expression (e.g. \\boxed{\\frac{1}{2}}).
    Returns None if no \\boxed{} is found.
    """
    match = re.search(r'\\boxed\s*\{', text)
    if not match:
        return None
    start = match.end() - 1  # position of the opening {
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison: strip space, commas, trailing punct, lowercase."""
    s = s.replace(' ', '').replace(',', '').lower()
    s = s.rstrip('.;:!?')
    return s


def is_correct(model_answer: str, reference_solution: str) -> bool:
    """Check if model answer matches the reference solution.

    Tries \\boxed{} extraction first. Falls back to last non-empty line.
    Both answers are normalized before comparison.
    """
    model_final = extract_boxed_answer(model_answer)
    if model_final is None:
        lines = [l.strip() for l in model_answer.strip().split('\n') if l.strip()]
        model_final = lines[-1] if lines else model_answer.strip()

    ref_final = extract_boxed_answer(reference_solution) or reference_solution.strip()
    return normalize_answer(model_final) == normalize_answer(ref_final)


# --- s1 Metrics ---

def compute_s1_metrics(points: list[dict]) -> dict:
    """Compute the three s1 paper metrics from evaluation points.

    Args:
        points: List of dicts with keys:
            config (str): Config name.
            x (float): Average thinking tokens for this config.
            y (float): Accuracy for this config.
            controlled (bool): Whether actual tokens stayed within budget * 1.1.

    Returns:
        dict with keys:
            control (float): Fraction of configs where controlled=True.
            control_pct (float): Same as control, as percentage.
            scaling (float): Average slope across adjacent (x, y) points.
            performance (float): Max accuracy across configs.
            points (list): Sorted points (same as input, sorted by x).
    """
    if not points:
        return {
            "control": 0.0,
            "control_pct": 0.0,
            "scaling": 0.0,
            "performance": 0.0,
            "points": [],
        }

    sorted_pts = sorted(points, key=lambda p: p["x"])

    # Control: fraction of configs where model stayed within budget * 1.1
    controlled_count = sum(1 for p in sorted_pts if p["controlled"])
    total = len(sorted_pts)
    control = controlled_count / total

    # Performance: max accuracy across configs
    performance = max(p["y"] for p in sorted_pts)

    # Scaling: average slope across adjacent accuracy-vs-thinking_tokens points
    slopes = []
    for i in range(len(sorted_pts) - 1):
        dx = sorted_pts[i + 1]["x"] - sorted_pts[i]["x"]
        dy = sorted_pts[i + 1]["y"] - sorted_pts[i]["y"]
        if dx > 0:
            slopes.append(dy / dx)
    scaling = sum(slopes) / len(slopes) if slopes else 0.0

    return {
        "control": control,
        "control_pct": control * 100.0,
        "scaling": scaling,
        "performance": performance,
        "points": sorted_pts,
    }


# --- Data loading ---

def load_eval_data(data_path: str, num_samples: int = 200, seed: int = 42) -> list[dict]:
    """Load evaluation questions from JSONL, deterministically subsample.

    Each line: {"question": "...", "solution": "..."}
    Uses a fixed seed for reproducibility.
    """
    questions = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if len(questions) > num_samples:
        rng = random.Random(seed)
        samples = rng.sample(questions, num_samples)
    else:
        samples = questions[:]

    return samples


# --- Model loading ---

def load_model_and_tokenizer(adapter_path: str | None = None, logger: logging.Logger | None = None):
    """Load Qwen2.5-7B-Instruct with 4-bit NF4 and optional LoRA adapter.

    Uses the same BitsAndBytesConfig as train.py. When adapter_path is None,
    loads the base model only (for baseline evaluation).
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    if logger:
        logger.info(f"Loading base model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    if adapter_path:
        if logger:
            logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        if logger:
            logger.info("LoRA adapter loaded successfully")

    return model, tokenizer


# --- Evaluation ---

def evaluate_config(controller, questions, name, max_thinking_tokens, num_suppressions, max_new_tokens, logger):
    """Evaluate a single BF config across all questions.

    Args:
        controller: BudgetForcingController instance.
        questions: List of {question, solution} dicts.
        name: Config name (for logging).
        max_thinking_tokens: Thinking token budget (0 = no budget).
        num_suppressions: How many times to suppress end-of-thinking.
        max_new_tokens: Max new tokens to generate.
        logger: Logger instance.

    Returns:
        List of per-question result dicts.
    """
    # max_thinking_tokens=0 means "no budget forcing" -> pass sentinel
    bf_max_tokens = max_thinking_tokens if max_thinking_tokens > 0 else NO_BUDGET_SENTINEL

    results = []
    for i, q_item in enumerate(tqdm(questions, desc=f"  [{name}]")):
        q = q_item["question"]
        solution = q_item.get("solution", "")
        prompt = EVAL_PROMPT_TEMPLATE.format(question=q)

        start_time = time.time()
        try:
            result = controller.generate(
                prompt=prompt,
                max_thinking_tokens=bf_max_tokens,
                num_suppressions=num_suppressions,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"  [{name}] question {i} failed: {e}")
            result = {
                "answer": "",
                "thinking_tokens": 0,
                "forced_end": False,
                "suppressions_used": 0,
            }
        elapsed = time.time() - start_time

        # If answer is empty (e.g. base model without s1 markers), fall back
        # to extracting from the full generation output.
        answer = result.get("answer", "").strip()
        if not answer:
            answer = result.get("full_output", "")
        correct = is_correct(answer, solution)

        results.append({
            "question_id": i,
            "question": (q[:120] + "...") if len(q) > 120 else q,
            "correct": correct,
            "thinking_tokens": result.get("thinking_tokens", 0),
            "forced_end": result.get("forced_end", False),
            "suppressions_used": result.get("suppressions_used", 0),
            "time_s": round(elapsed, 2),
            "answer": (result.get("answer", "") or "")[:200],
            "config": name,
        })

    return results


def _find_config_max_think(name: str) -> int:
    """Look up the max_thinking_tokens setting for a given config name."""
    for cfg_name, mt, _ns, _mn in BF_CONFIGS + BASELINE_CONFIGS:
        if cfg_name == name:
            return mt
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-config evaluation with s1 metrics"
    )
    parser.add_argument("--adapter", required=True,
                        help="Path to trained LoRA adapter")
    parser.add_argument("--eval_data", required=True,
                        help="Path to eval questions JSONL")
    parser.add_argument("--results_dir", default="/content/s1-t4/results",
                        help="Output directory for results (default: /content/s1-t4/results)")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip base model evaluation (adapter + BF configs only)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of eval questions to use (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question selection (default: 42)")
    args = parser.parse_args()

    # Setup directories and logging
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = setup_logging(LOG_DIR)
    logger.info(f"Arguments: {vars(args)}")

    # Load evaluation data
    logger.info(f"Loading eval data from {args.eval_data}...")
    questions = load_eval_data(args.eval_data, args.num_samples, args.seed)
    logger.info(f"Loaded {len(questions)} eval questions")

    if len(questions) == 0:
        logger.error("No evaluation questions loaded!")
        sys.exit(1)

    # Load adapter model
    from budget_forcing import BudgetForcingController

    logger.info("Loading adapter model...")
    adapter_model, tokenizer = load_model_and_tokenizer(args.adapter, logger)
    adapter_controller = BudgetForcingController(adapter_model, tokenizer)

    # Evaluate adapter with BF configs
    all_results = []
    for name, max_think, num_supp, max_new in BF_CONFIGS:
        logger.info(f"Evaluating config: {name} "
                     f"(max_think={max_think}, supp={num_supp}, max_new={max_new})")
        results = evaluate_config(
            adapter_controller, questions, name,
            max_think, num_supp, max_new, logger,
        )
        all_results.extend(results)
        n_correct = sum(r["correct"] for r in results)
        logger.info(f"  [{name}] accuracy: {n_correct}/{len(results)} "
                     f"({n_correct / len(results) * 100:.1f}%)")

    # Evaluate baselines (base model, no adapter)
    if not args.skip_baselines:
        logger.info("Loading base model (no adapter) for baselines...")
        base_model, _ = load_model_and_tokenizer(None, logger)
        base_controller = BudgetForcingController(base_model, tokenizer)

        for name, max_think, num_supp, max_new in BASELINE_CONFIGS:
            logger.info(f"Evaluating baseline: {name} "
                         f"(max_think={max_think}, supp={num_supp}, max_new={max_new})")
            results = evaluate_config(
                base_controller, questions, name,
                max_think, num_supp, max_new, logger,
            )
            all_results.extend(results)
            n_correct = sum(r["correct"] for r in results)
            logger.info(f"  [{name}] accuracy: {n_correct}/{len(results)} "
                         f"({n_correct / len(results) * 100:.1f}%)")
    else:
        logger.info("Skipping baseline evaluation (--skip_baselines)")

    # --- Aggregate results ---

    # Build config-level summaries
    configs: dict[str, dict] = {}
    for r in all_results:
        cfg = r["config"]
        if cfg not in configs:
            configs[cfg] = {"correct": 0, "total": 0, "thinking_tokens": []}
        configs[cfg]["correct"] += 1 if r["correct"] else 0
        configs[cfg]["total"] += 1
        configs[cfg]["thinking_tokens"].append(r["thinking_tokens"])

    config_summary = {}
    for cfg_name, data in configs.items():
        avg_tokens = (
            sum(data["thinking_tokens"]) / len(data["thinking_tokens"])
            if data["thinking_tokens"] else 0.0
        )
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        config_summary[cfg_name] = {
            "accuracy": round(acc, 4),
            "accuracy_pct": round(acc * 100, 1),
            "correct": data["correct"],
            "total": data["total"],
            "avg_thinking_tokens": round(avg_tokens, 1),
        }

    # Build points for s1 metrics
    points = []
    for cfg_name, summary in config_summary.items():
        max_think = _find_config_max_think(cfg_name)
        avg_tokens = summary["avg_thinking_tokens"]

        # Tag as "sft" or "baseline" so charts.py doesn't rely on naming prefix
        is_baseline = any(cfg_name == bc[0] for bc in BASELINE_CONFIGS)
        point_type = "baseline" if is_baseline else "sft"

        # Controlled if actual avg tokens <= max_thinking_setting * 1.1
        # Configs with max_think=0 (no budget) are always considered controlled
        if max_think == 0:
            controlled = True
        else:
            controlled = avg_tokens <= max_think * 1.1

        points.append({
            "config": cfg_name,
            "type": point_type,
            "x": avg_tokens,
            "y": summary["accuracy"],
            "controlled": controlled,
        })

    # Compute s1 metrics
    metrics = compute_s1_metrics(points)
    logger.info(
        f"s1 Metrics: control={metrics['control_pct']:.1f}%, "
        f"scaling={metrics['scaling']:.6f}, "
        f"performance={metrics['performance']:.4f}"
    )

    # --- Write output files ---

    # metrics.json
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # per_question.jsonl
    per_q_path = os.path.join(args.results_dir, "per_question.jsonl")
    with open(per_q_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved per-question results ({len(all_results)} lines) to {per_q_path}")

    # config_summary.json
    cfg_path = os.path.join(args.results_dir, "config_summary.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "configs": config_summary,
            "n_questions": len(questions),
        }, f, indent=2)
    logger.info(f"Saved config summary to {cfg_path}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
