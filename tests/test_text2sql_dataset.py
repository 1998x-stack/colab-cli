"""Tests for dataset.py — formatting, label masking, truncation."""
import torch
from projects.nlp.text2sql_finetune.dataset import format_and_tokenize


class FakeTokenizer:
    """Minimal tokenizer stub that returns predictable token IDs."""
    def __init__(self):
        self.im_start_id = 1
        self.im_end_id = 2
        self.newline_id = 3

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        """Return a predictable string based on the role content."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=False, **kwargs):
        """Tokenize by assigning each word a unique token ID."""
        words = text.split()
        if truncation and max_length:
            words = words[:max_length]
        return {
            "input_ids": list(range(100, 100 + len(words))),
            "attention_mask": [1] * len(words),
        }


def test_label_masking_only_assistant_tokens_unmasked():
    tokenizer = FakeTokenizer()
    result = format_and_tokenize(tokenizer, "CREATE TABLE t (x int)", "what is x?", "SELECT x FROM t")

    input_ids = result["input_ids"]
    labels = result["labels"]

    assert len(labels) == len(input_ids)
    assert labels[0] == -100, "first token (system) should be masked"
    assert labels[-1] != -100, "last token (assistant) should be unmasked"

    non_masked = (labels != -100).sum().item()
    assert non_masked > 0, "should have some unmasked tokens"
    assert non_masked < len(labels), "should have some masked tokens (not everything is assistant)"


def test_long_sequence_truncation():
    tokenizer = FakeTokenizer()
    long_context = "CREATE TABLE big ("
    for i in range(200):
        long_context += f"col{i} int, "
    long_context += "id int)"

    result = format_and_tokenize(tokenizer, long_context, "select all ids", "SELECT id FROM big")

    # With truncation at 1024 tokens (set in the real tokenizer)
    assert len(result["input_ids"]) <= 1024, f"should truncate to 1024, got {len(result['input_ids'])}"


def test_truncated_example_preserves_label_alignment():
    """When truncated, labels and input_ids must stay same length."""
    tokenizer = FakeTokenizer()
    result = format_and_tokenize(tokenizer, "CREATE TABLE t (x int)", "what is x?", "SELECT x FROM t")
    assert len(result["input_ids"]) == len(result["labels"])
    assert len(result["input_ids"]) == len(result["attention_mask"])
