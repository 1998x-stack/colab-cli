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


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate stats from per-example results."""
    n = len(results)
    em_sum = sum(r["em"] for r in results)
    f1_sum = sum(r["f1"] for r in results)
    latencies = [r["latency_s"] for r in results if r.get("latency_s")]
    tokens = [r["tokens"] for r in results]

    return {
        "exact_match": round(em_sum / n, 4) if n else 0,
        "f1": round(f1_sum / n, 4) if n else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "total_tokens": sum(tokens),
        "avg_tokens_per_example": round(sum(tokens) / n, 1) if n else 0,
    }


def compute_all(cot_results: list[dict], react_results: list[dict]) -> dict:
    """Join CoT and ReAct results, compute per-example metrics, and aggregate."""
    react_by_id = {r["id"]: r for r in react_results}

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
            "cot_latency_s": cr["latency_s"],
            "cot_tokens": cr["tokens"],
            "react_prediction": rr["prediction"],
            "react_em": exact_match(rr["prediction"], rr["answer"]),
            "react_f1": round(f1_score(rr["prediction"], rr["answer"]), 4),
            "react_latency_s": rr["latency_s"],
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

    cot_lat = [r["cot_latency_s"] for r in per_example if r["cot_latency_s"]]
    react_lat = [r["react_latency_s"] for r in per_example if r["react_latency_s"]]

    return {
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "dataset": "hotpot_qa",
            "n_examples": len(per_example),
        },
        "cot": {
            "exact_match": round(sum(cot_em) / len(cot_em), 4),
            "f1": round(sum(cot_f1) / len(cot_f1), 4),
            "avg_latency_s": round(sum(cot_lat) / len(cot_lat), 3) if cot_lat else 0,
            "total_tokens": sum(r["cot_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["cot_tokens"] for r in per_example) / len(per_example), 1),
        },
        "react": {
            "exact_match": round(sum(react_em) / len(react_em), 4),
            "f1": round(sum(react_f1) / len(react_f1), 4),
            "avg_latency_s": round(sum(react_lat) / len(react_lat), 3) if react_lat else 0,
            "total_tokens": sum(r["react_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["react_tokens"] for r in per_example) / len(per_example), 1),
            "avg_steps": round(sum(react_steps) / len(react_steps), 1),
            "step_distribution": step_dist,
        },
        "per_example": per_example,
    }
