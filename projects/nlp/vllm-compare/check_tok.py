import transformers
import tokenizers
print(f"transformers: {transformers.__version__}")
print(f"tokenizers: {tokenizers.__version__}")

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
print(f"Type: {type(t).__name__}")
attr = "all_special_tokens_extended"
print(f"Has {attr}: {hasattr(t, attr)}")
