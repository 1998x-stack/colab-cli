# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase map

```
projects/
├── alexnet_imagenette/   # AlexNet faithful reproduction (Imagenette, 10-class)
│   ├── alexnet.py         # Paper arch + He init, build_alexnet(config) factory
│   ├── train.py           # ImageFolder pipeline, PCA aug, training, 10-view eval, 4-expt orchestrator, charts
│   ├── launch.py          # Colab bootstrap: pip install, spawn train+watchdog detached
│   ├── watchdog.py        # Writes /content/heartbeat.json every 30s
│   └── check_progress.py  # Reads heartbeat, pgrep, tail log, health alerts
├── rl-sac/               # SAC on MountainCarContinuous
├── cnn-cifar10/           # CNN classifier, CIFAR-10
├── rl-dqn-atari/ nanogpt/ nanochat-colab/ rnn-imdb/ cuda-tutorial/
├── vllm-compare/        # vLLM model benchmarks on Colab T4
│   ├── compare.py         # 3-model latency/throughput/VRAM comparison
│   ├── test_v0.py         # Working config: V0 engine + monkey-patch for T4
│   ├── bootstrap.py       # Colab bootstrap (pip install + spawn)
│   └── check_progress.py
├── vllm-rag/ ml-tutorial/ autoresearch-t4/
├── transformer_iwslt/    # Transformer (Attention Is All You Need) on IWSLT'14 De->En
│   ├── model.py           # Encoder-decoder Transformer, 65M params, 3 configs
│   ├── train.py           # IWSLT data pipeline, BPE tokenizer, training loop, beam search
│   ├── launch.py          # Colab bootstrap with checkpoint-resume
│   ├── check_progress.py  # Cron-based training monitor
│   ├── checkpoint.py      # Save/load helpers for multi-session resume
│   └── charts.py          # Post-hoc charts (loss, BLEU, ablation, attention, PE)
```

## Project conventions

- **Dir naming**: `snake_case` for Python imports (`alexnet_imagenette`, not hyphens)
- **File pattern**: `train.py` + `launch.py` (bootstrap) + `watchdog.py` (heartbeat) + `check_progress.py` (monitor)
- **Multi-experiment**: Upload `exp_ids.txt` per session; launch.py reads it, passes `--exp_ids` to train.py
- **Output**: VM `/content/<project>-output/` → download to `projects/<project>/output/`

## Accounts & proxy

```bash
# colab (hackxie1998) — default, proxy via Clash
colab new --gpu T4 -s <name> && colab exec -s <name> -f script.py --timeout 120

# cc (xbetterdetermine) — often needs no_proxy for WebSocket
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
HOME=~/colab-accounts/account-c colab new --gpu T4 -s <name>
HOME=~/colab-accounts/account-c colab exec -s <name> -f script.py --timeout 120

# cb (stefaniehu929) — same pattern

# clb (xieminghack) — same pattern
```

Only 1 GPU per free account. SSL errors transient — retry before assuming dead.

Free-tier GPU (T4) sessions last ~12-15 min — keep experiments under 10 min total.

## Free-tier reality

**Sessions die in 8-12 min on T4.** Keep experiments under ~4 min each. 20-epoch AlexNet/Imagenette ≈ 3.5 min. Parallelize across accounts. Download immediately on completion.

## Core workflow

```bash
# 1. Provision + upload + launch
colab new --gpu T4 -s training
colab upload *.py /content/
colab exec -s training -f launch.py --timeout 120   # detaches train subprocess

# 2. Monitor (local cron + VM watchdog)
CronCreate cron="*/5 * * * *" prompt="Check session..." durable=true recurring=true

# 3. Download + cleanup
colab download /content/<project>-output.tar.gz projects/<project>/output/
colab stop -s training
```

## Detached training (gotchas)

- `PYTHONUNBUFFERED=1` + `python -u` + `start_new_session=True` in subprocess.Popen
- `colab exec -f` reads LOCAL files (relative paths), sends to VM. No `-c` flag — use stdin pipe.
- `colab download` needs tar for dirs: `tar -czf /content/out.tar.gz -C /content dir/`

## vLLM on Colab

**Only vLLM 0.10.2 fits** Colab T4 (CUDA 12.8, 15.6 GB VRAM). Newer versions (0.21.0+) need CUDA 13.0.

Two required workarounds:

1. **`VLLM_USE_V1=0`** — Colab pre-initializes CUDA, so V1's `spawn` subprocess crashes. Must set before importing vllm.

2. **Transformers 5.x monkey-patch** — vLLM 0.10.2 pins `transformers>=5.0` but the code still uses `all_special_tokens_extended` (removed in 5.x).

```python
import os
os.environ["VLLM_USE_V1"] = "0"

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
_orig_init = PreTrainedTokenizerBase.__init__
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    if not hasattr(self, "all_special_tokens_extended"):
        self.all_special_tokens_extended = []
PreTrainedTokenizerBase.__init__ = _patched_init

# THEN import vllm
from vllm import LLM, SamplingParams
```

Install: `pip install vllm==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128`

After install, reinstall torchvision (vLLM downgrades torch → breaks pre-installed torchvision):
`pip install torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cu128`

VRAM fit (T4 15.6 GB): SmolLM2-1.7B ~12.8 GB. Qwen2.5-3B likely fits. 7B needs AWQ quantization.

## Architecture gotchas (from AlexNet project)

- **Paper init fails on 128×128 input**: `N(0,0.01)` → loss stuck at ln(10). Use He init + `clip_grad_norm_(5.0)` + LR=0.001
- **PCA resize**: Must use `TF.resize(img, [H, W])` (fixed size), not `TF.resize(img, H)` (preserves aspect ratio → variable tensor sizes)
- **HF datasets unreliable on Colab**: `datasets` version too new for older dataset scripts. Prefer `torchvision.datasets.ImageFolder` + direct download
- **Data aug hurts at low epochs**: Without 90+ epochs, augmentation just slows convergence. "No Data Aug" beats baseline at 20 epochs — expected.
- **10-view eval**: Paper protocol — 4 corners + center, each flipped. Average softmax across views before computing accuracy.

## Pre-deploy checklist (avoid the #1 bug pattern)

- Run forward pass locally (random tensor) to verify model output shape and no NaN
- Fit PCA on a sample locally to verify resize → stack doesn't crash
- Validate data pipeline loads images correctly (check first batch shapes + labels)
- `grep` the codebase for `load_dataset` — if found, verify HF dataset name still works on Colab's `datasets` version

## Doc protocol

- After writing docs/charts, open them for review. Don't claim "done" without visual verification.
- Write gotchas.md per-project proactively — don't wait to be asked.

## Multi-session awareness

- User runs parallel Claude Code sessions (41% of messages). Files may change mid-session.
- Skills live in `.claude/skills/<name>/SKILL.md` — not `.agents/skills/` or other paths.
