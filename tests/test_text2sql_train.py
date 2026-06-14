"""Tests for train.py — forward pass, loss behavior. Uses tiny synthetic model (no GPU needed)."""
import torch
from transformers import AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model

# Use a tiny 2-layer config for fast local testing
TINY_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "vocab_size": 1000,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def create_tiny_model():
    config = LlamaConfig(**TINY_CONFIG)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    return model


def create_tiny_batch(batch_size=2, seq_len=64):
    """Returns a batch of tokenized data with some labels unmasked."""
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Last 20 tokens are "assistant" — unmasked
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[:, -20:] = input_ids[:, -20:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_forward_pass_no_nan():
    model = create_tiny_model()
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    batch = create_tiny_batch()
    outputs = model(**batch)
    loss = outputs.loss

    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"


def test_loss_decreases_after_step():
    model = create_tiny_model()
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    batch = create_tiny_batch()

    model.train()
    loss1 = model(**batch).loss
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss2 = model(**batch).loss
    assert loss2.item() < loss1.item(), f"Loss did not decrease: {loss2.item():.4f} >= {loss1.item():.4f}"
