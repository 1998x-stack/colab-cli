"""Load b-mc2/sql-create-context, format into Qwen3 chat template, save as .pt tensors.

Usage:
    python dataset.py --split train --max_examples 500 --output data/train.pt
    python dataset.py --split auto --train_examples 500 --test_examples 100 --train_output data/train.pt --test_output data/test.pt

Note: b-mc2/sql-create-context only has a 'train' split. Use --split auto to
automatically load all data and split into train/test.
"""
import argparse
import os
import torch
from datasets import load_dataset, get_dataset_split_names
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


def process_dataset(tokenizer, dataset, max_examples, desc=""):
    examples = []
    skipped = 0
    for row in dataset:
        context = row.get("context", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not context or not question or not answer:
            skipped += 1
            continue
        tokenized = format_and_tokenize(tokenizer, context, question, answer)
        if (tokenized["labels"] == -100).all():
            skipped += 1
            continue
        examples.append(tokenized)
        if max_examples > 0 and len(examples) >= max_examples:
            break
    if skipped:
        print(f"[{desc}] Skipped {skipped} examples (empty fields or prompt too long)")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        help="Dataset split name, or 'auto' to load all data and split")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--output", help="Output path (required for non-auto mode)")
    parser.add_argument("--train_examples", type=int, default=500)
    parser.add_argument("--test_examples", type=int, default=100)
    parser.add_argument("--train_output", default="data/train.pt")
    parser.add_argument("--test_output", default="data/test.pt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if args.split == "auto":
        # Load all available data and split into train/test
        available = get_dataset_split_names("b-mc2/sql-create-context")
        print(f"Available splits: {available}")
        # Load from the first available split (usually only 'train')
        split_name = available[0]
        dataset = load_dataset("b-mc2/sql-create-context", split=split_name)
        total = len(dataset)
        print(f"Loaded {total} examples from '{split_name}' split")

        # Ensure we have enough data
        needed = args.train_examples + args.test_examples
        if total < needed:
            print(f"WARNING: dataset has {total} examples, requested {needed}. Using all available.")
            args.train_examples = int(total * 0.85)
            args.test_examples = total - args.train_examples

        all_data = list(dataset)
        train_data = all_data[:args.train_examples]
        test_data = all_data[args.train_examples:args.train_examples + args.test_examples]

        train_examples = process_dataset(tokenizer, train_data, args.train_examples, desc="train")
        test_examples = process_dataset(tokenizer, test_data, args.test_examples, desc="test")

        os.makedirs(os.path.dirname(args.train_output) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(args.test_output) or ".", exist_ok=True)
        torch.save(train_examples, args.train_output)
        torch.save(test_examples, args.test_output)
        print(f"Saved {len(train_examples)} train → {args.train_output}")
        print(f"Saved {len(test_examples)} test → {args.test_output}")
    else:
        # Explicit split mode (backward compatible)
        available = get_dataset_split_names("b-mc2/sql-create-context")
        if args.split not in available:
            print(f"ERROR: split '{args.split}' not found. Available: {available}")
            print(f"Use --split auto to auto-split from available data, or specify one of: {available}")
            raise SystemExit(1)

        if not args.output:
            parser.error("--output is required for non-auto mode")

        dataset = load_dataset("b-mc2/sql-create-context", split=args.split)
        if args.max_examples > 0:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))

        examples = process_dataset(tokenizer, dataset, args.max_examples, desc=args.split)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(examples, args.output)
        print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
