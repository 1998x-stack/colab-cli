"""EM and F1 metrics for HotpotQA evaluation."""
import re
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove articles/punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return 1 if normalize(prediction) == normalize(ground_truth) else 0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    n_common = sum(common.values())

    precision = n_common / len(pred_tokens) if pred_tokens else 0
    recall = n_common / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all(cot_results: list[dict], react_results: list[dict],
                cot_time_s: float = 0, react_time_s: float = 0) -> dict:
    """Join CoT and ReAct results, compute per-example metrics, and aggregate.

    cot_time_s and react_time_s are the total wall-clock seconds for each
    strategy. Per-example amortized latency is derived from these.
    """
    n = len(cot_results)
    react_by_id = {r["id"]: r for r in react_results}

    cot_per_amortized = cot_time_s / n if n else 0
    react_per_amortized = react_time_s / n if n else 0

    per_example = []
    for cr in cot_results:
        rr = react_by_id[cr["id"]]
        per_example.append({
            "id": cr["id"],
            "question": cr["question"],
            "answer": cr["answer"],
            "cot_prediction": cr["prediction"],
            "cot_em": exact_match(cr["prediction"], cr["answer"]),
            "cot_f1": round(f1_score(cr["prediction"], cr["answer"]), 4),
            "cot_tokens": cr["tokens"],
            "react_prediction": rr["prediction"],
            "react_em": exact_match(rr["prediction"], rr["answer"]),
            "react_f1": round(f1_score(rr["prediction"], rr["answer"]), 4),
            "react_tokens": rr["tokens"],
            "react_steps": rr.get("steps", 0),
        })

    cot_em = [r["cot_em"] for r in per_example]
    cot_f1 = [r["cot_f1"] for r in per_example]
    react_em = [r["react_em"] for r in per_example]
    react_f1 = [r["react_f1"] for r in per_example]

    react_steps = [r["react_steps"] for r in per_example]
    step_dist = {}
    for s in react_steps:
        step_dist[str(s)] = step_dist.get(str(s), 0) + 1

    return {
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "dataset": "hotpot_qa",
            "n_examples": n,
        },
        "cot": {
            "exact_match": round(sum(cot_em) / len(cot_em), 4),
            "f1": round(sum(cot_f1) / len(cot_f1), 4),
            "avg_latency_s": round(cot_per_amortized, 3),
            "total_wall_time_s": round(cot_time_s, 1),
            "total_tokens": sum(r["cot_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["cot_tokens"] for r in per_example) / n, 1) if n else 0,
        },
        "react": {
            "exact_match": round(sum(react_em) / len(react_em), 4),
            "f1": round(sum(react_f1) / len(react_f1), 4),
            "avg_latency_s": round(react_per_amortized, 3),
            "total_wall_time_s": round(react_time_s, 1),
            "total_tokens": sum(r["react_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["react_tokens"] for r in per_example) / n, 1) if n else 0,
            "avg_steps": round(sum(react_steps) / len(react_steps), 1),
            "step_distribution": step_dist,
        },
        "per_example": per_example,
    }
