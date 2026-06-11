# s1-t4 Gotchas

Field-tested surprises specific to the s1 test-time scaling project.

## Colab deployment

### HF model download: "0%" progress doesn't mean stuck

When downloading Qwen2.5-7B-Instruct (~15GB) on Colab, the transformers progress bar shows `Fetching 4 files: 0%| 0/4` for many minutes. The download IS progressing — check `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/` for `.incomplete` files. The progress bar only updates when a file completes (each shard is ~2-5GB).

On Colab T4, the 15GB download takes ~15-25 minutes through the VM's network.

### Session dies silently during long downloads

Both T4 sessions (accounts cb and cc) died during the 15GB model download phase. This was NOT the WebSocket disconnection — the session itself terminated (`colab sessions` returned "No active sessions"). Root cause unclear — possibly Google's free-tier preemption during sustained download activity, or GPU quota enforcement.

**Mitigation:** Pre-download model weights to Google Drive and mount with `colab drivemount`. The VM→Drive path uses Google's internal network and bypasses both the China proxy AND HF download bandwidth.

### WebSocket exec drops while setup-download runs inline

Trying to combine `pip install` + `dataset.py` data generation + `train.py` launch into a single `colab exec` call fails because model download within the exec session takes >12 minutes. The WebSocket drops before the download completes.

**Fix:** Always use the two-phase bootstrap pattern:
1. Exec a short script that ONLY spawns the real workload detached
2. The detached workload (train.py) handles download + training

## Data

### s1K field name: `thinking_trajectories` not `reasoning_trace`

The HuggingFace dataset `simplescaling/s1K` uses column name `thinking_trajectories` (a list, always 1 element). The paper and early code used `reasoning_trace`. The raw data does NOT contain `<|im_start|>think` markers — those are added by `format_sample()`.

### MATH500 requires pre-downloaded data

The `datasets` library is not installed locally. Loading MATH500 requires either:
- Installing `datasets` locally (creates version conflicts with other projects)
- Downloading MATH500 JSON from HF CDN directly
- Using s1K data with `\boxed{}` answers as a proxy eval set (153/300 of s1K filtered samples qualify)

For serious evaluation, pre-download MATH500 once and save as a local JSONL file.

### `--skip-difficulty` uses trace length proxy

When running dataset.py with `--skip-difficulty` (to avoid loading Qwen2.5-7B on CPU), the difficulty filter uses `trace_len > median` as a proxy for question difficulty. This keeps ~50% of samples. The full difficulty filter (evaluating each question with the base model) is more selective but requires GPU.

## QLoRA training

### max_seq_length=4096 cuts off ~26% of s1K samples

s1K average reasoning trace is ~4,700 tokens (Qwen tokenizer). With max_seq_length=4096, about 26% of samples are truncated. The paper used 32768 to avoid truncation but this requires more VRAM. On T4 with QLoRA, 4096 is the practical limit.

### Training on T4: 2-3 hours for 300 samples × 3 epochs

Actual measured: model download + tokenizer load = ~20 min, training 300 samples × 3 epochs × batch=16 ≈ 2.5 hours. Total session time ~3 hours. A 12h Colab free session is sufficient.

### Loss masking: mask everything before `<|im_start|>assistant`

The `S1KDataset._mask_prefix` finds the `<|im_start|>assistant` token sequence in the input_ids and masks everything from position 0 through the end of that marker. This means the assistant marker tokens themselves are also masked (not just "before" them). This is fine — the tokens right after the marker are the think section which is unmasked.

## Budget Forcing

### "Wait" suppression vs. "Wait" appending

The s1 paper's Budget Forcing extends thinking by TWO mechanisms: (a) suppressing the end-of-thinking token, AND (b) appending "Wait" to the model's context. Our implementation does (a) via logit manipulation — when the model tries to output the end-of-thinking token, we suppress it so it picks the second-best token and continues. The "Wait" text is NOT actually appended to the context.

This is documented in `budget_forcing.py` docstring. The behavior is different from the paper but achieves the same goal through a different mechanism (logit-level suppression vs. text-level prompting).

### Base model (no LoRA) always returns 0% with s1 markers

The `BudgetForcingController._extract_answer()` searches for `<|im_start|>answer` markers. The base Qwen2.5-7B-Instruct (not fine-tuned on s1K) never produces these markers, so extracted answers are empty. The eval code falls back to `extract_boxed_answer()` on the full output.

### BudgetForcingLogitsProcessor requires batch_size=1

The LogitsProcessor hardcodes `scores[0, ...]`. Any attempt to generate with batch_size > 1 raises `ValueError`. This is enforced by a guard in `__call__`.
