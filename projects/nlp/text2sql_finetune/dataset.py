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
        # Skip if prompt fills entire context window (no room for answer)
        if (tokenized["labels"] == -100).all():
            continue
        examples.append(tokenized)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(examples, args.output)
    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
