# s1-T4

QLoRA fine-tuning of Qwen2.5-7B-Instruct on a filtered subset of the s1K dataset (test-time training scaling) on a Colab T4 GPU, using HuggingFace transformers, PEFT, and bitsandbytes.

## Usage

```bash
# Local training (requires 4-bit capable GPU with >=16GB VRAM)
python train.py --data s1k_filtered.jsonl

# Colab deployment (see ../.claude/skills/colab-cli/SKILL.md)
cb launch.py
```

## Key results

No training results available in this repository. The project is set up for training but has not been run to completion.

Expected setup:
| Setting | Value |
|---------|-------|
| Base model | Qwen2.5-7B-Instruct (4-bit NF4 quantized) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable params | ~0.3% of total |
| Effective batch size | 16 (2 per device x 8 grad accum) |
| Max sequence length | 4096 |
| Dataset | ~300 filtered s1K samples |
| Total training time (est.) | ~3 hours on T4 |
| Precision | bf16 |

## Gotchas

- Qwen2.5-7B-Instruct download (~15GB) takes 15-25 minutes on Colab T4. The progress bar shows `0%` for minutes at a time because it only updates when each shard file completes.
- Sessions can die silently during the long model download phase. Mitigation: pre-download to Google Drive and mount with `colab drivemount`.
- `max_seq_length=4096` truncates ~26% of s1K samples (average reasoning trace is ~4700 tokens). The paper used 32768.
- WebSocket exec drops if setup/training takes >12 min. Always use the two-phase bootstrap pattern: the launch script spawns training as a detached subprocess, then exits.
- Budget Forcing in this implementation suppresses the end-of-thinking token via logit manipulation, but does NOT append "Wait" to the context (differing from the s1 paper).
- The base Qwen2.5-7B-Instruct model (no LoRA fine-tuning) never produces `<|im_start|>answer` markers, so `_extract_answer()` falls back to `extract_boxed_answer()`.
- BudgetForcingLogitsProcessor requires batch_size=1.
- Loss masking hides tokens before `<|im_start|>assistant` (including the marker itself), focusing learning on the assistant's reasoning and answer.
- Data field is `thinking_trajectories` (a list), not `reasoning_trace` as the paper originally used.
