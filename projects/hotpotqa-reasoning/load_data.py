"""Download HotpotQA distractor set, sample 200 examples, save to JSON."""
import json
import os
import random

OUTPUT_PATH = "/content/data.json"
N_SAMPLES = 200
SEED = 42


def load_and_sample() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=False)
    print(f"Loaded HotpotQA distractor validation: {len(ds)} examples")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(ds)), N_SAMPLES)
    examples = []
    for i in indices:
        row = ds[int(i)]
        examples.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],          # "comparison" or "bridge"
            "level": row["level"],        # "easy", "medium", "hard"
            "context": _flatten_context(row["context"]),
        })
    print(f"Sampled {len(examples)} examples")
    return examples


def _flatten_context(context: dict) -> str:
    """Format context passages into a single string with titles."""
    parts = []
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    for i, (title, sents) in enumerate(zip(titles, sentences)):
        parts.append(f"[{i}] {title}: {' '.join(sents)}")
    return "\n\n".join(parts)


def save(examples: list[dict], path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":
    examples = load_and_sample()
    save(examples)
