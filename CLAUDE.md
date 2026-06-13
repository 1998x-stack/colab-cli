# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills-first workflow

This project has two task-specific skills that handle Colab and Kaggle operations:

| Skill | Trigger | Covers |
|-------|---------|--------|
| **colab-cli** | Colab, GPU VM, `colab` commands, remote training | Provision, exec, monitor, multi-account, all gotchas |
| **kaggle-cli** | Kaggle, `kaggle` CLI, kernel push, GPU notebooks | Push, monitor, download, multi-account, GPU compatibility |

**When a task involves Colab or Kaggle, invoke the relevant skill via the Skill tool before acting.** The skill files are at `.claude/skills/<name>/SKILL.md` with supporting scripts and references.

The sections below provide project-specific context and constraints that the skills don't cover — account inventory, proxy setup, bash scripting rules, and domain-specific gotchas.

## Codebase map

```
.claude/skills/
├── colab-cli/             # Colab GPU VM management from terminal
│   ├── SKILL.md             # Full workflow: provision, exec, monitor, multi-account
│   ├── references/          # gotchas.md (22 items), workflows.md
│   └── scripts/             # launch.py, check_progress.py, launch_proxy.py
└── kaggle-cli/            # Kaggle Notebooks GPU training (push model, REST API)
    ├── SKILL.md             # Full workflow: push, monitor, download, multi-account
    ├── references/          # gotchas.md (16 items)
    └── scripts/             # push_and_wait.py, check_progress.py, kernel-metadata.json

projects/
├── alexnet_imagenette/   # AlexNet faithful reproduction (Imagenette, 10-class)
├── transformer_iwslt/    # Transformer (Attention Is All You Need) on IWSLT'14 De->En
├── vllm-compare/         # vLLM model benchmarks on Colab T4
├── vit-cifar10/          # ViT on CIFAR-10 (Kaggle, 3-config experiment)
├── rl-sac/ cnn-cifar10/ rl-dqn-atari/ nanogpt/ nanochat-colab/ rnn-imdb/ cuda-tutorial/
└── vllm-rag/ ml-tutorial/ autoresearch-t4/
```

## Project conventions

- **Dir naming**: `snake_case` for Python imports (`alexnet_imagenette`, not hyphens)
- **File pattern**: `train.py` + `launch.py` (bootstrap) + `watchdog.py` (heartbeat) + `check_progress.py` (monitor)
- **Multi-experiment**: Upload `exp_ids.txt` per session; launch.py reads it, passes `--exp_ids` to train.py
- **Output**: VM `/content/<project>-output/` → download to `projects/<project>/output/`

## Accounts

**Colab (4 accounts)** — isolated via `$HOME` directories. See colab-cli skill for full details.

```bash
colab   # hackxie1998 (default)    cb    # stefaniehu929
cc      # xbetterdetermine          clb   # xieminghack
```

**Kaggle (4 accounts)** — tokens in `.kaggle/access_token{1,2,3,4}`. Token 4 (xieming1998) is active. See kaggle-cli skill for multi-account management.

```bash
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./project
```

## Proxy setup (REQUIRED from China)

Colab uses two separate network paths with different proxy behavior. See colab-cli skill for full explanation. The essential two-config flip:

```bash
# Config A — REST through SOCKS5, WebSocket direct (try first):
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

# Config B — both through proxy, HTTP CONNECT tunnel (flip if A fails):
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

Which works changes per session — flip and retry.

**Diagnosing GPU provisioning failures (2026-06-13):**

| Config | Error on `colab new --gpu T4` | Meaning |
|--------|-------------------------------|---------|
| A (SOCKS5 + no_proxy) | `Service Unavailable` (503) | Ambiguous — could be proxy issue OR GPU exhaustion |
| B (HTTP CONNECT) | `Precondition Failed` (412) / `TooManyAssignmentsError` | Genuine GPU quota exhaustion |

**Procedure:** If config A returns 503, flip to config B. If config B returns 412, GPU is genuinely exhausted — try other accounts (`cb`, `cc`, `clb`). If ALL accounts return 412, GPU is globally unavailable — use CPU fallback (omit `--gpu`). CPU is fine for lightweight workloads (tabular RL, small models, debugging).

Config B handles the full workflow (new, upload, exec, download) without switching — when in doubt, start with B.

REST operations (`colab new`, `colab stop`, `colab sessions`, `colab download`) always use the proxy. Only `colab exec`/`colab upload` might need `no_proxy`.

**SOCKS5 needs PySocks for REST:** `pip install requests[socks]` or use `http://` style for REST.

**`colab upload` goes through WebSocket** — it fails when exec WebSocket is unstable. For multi-file projects, use the base64 embed pattern (see colab-cli gotchas.md).

## Key constraints (not in skills)

These are project-specific or hyper-specific operational constraints:

- **Checkpoint downloads >600MB fail through proxy**: Full checkpoint with optimizer state = ~1GB, proxy breaks at ~624MB (IncompleteRead). Save a separate **weights-only checkpoint** (~120-233MB) for download.
- **BLEU/beam search is the hidden bottleneck** (transformer_iwslt): Beam search eval on full val set takes hours. Use 100-sentence subset with greedy decode for training-time eval.
- **CUDA OOM during eval even when training fits** (transformer_iwslt): Beam search allocates extra tensors. Use `torch.cuda.empty_cache()` before eval. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **First Colab session rarely produces useful training**: Data download + tokenizer training + CUDA JIT = 7-10 min overhead. Combined with ~12-15 min effective exec window, first session dies before completing an epoch. Second session (data cached on VM) works normally.
- **Kaggle log streaming buffers**: GPU+internet kernels may show zero logs until completion. A 37-min P100 run produced all 105 log lines atomically. Empty logs ≠ stuck. See kaggle-cli skill monitoring section.

## Cron watchtower for long-running Colab jobs

Colab's WebSocket is unreliable from China — `colab exec` drops after 12-15 min, logs buffer, and you can't watch training live. **Set up a cron watchtower** that periodically fetches outputs from the VM via REST (`colab download`), which survives WebSocket drops.

**Pattern (2026-06-13, rl-sarsa-gym):**

1. Training writes all artifacts to a single output directory (`/content/<project>-output/`)
2. A local `fetch.sh` tars on VM → downloads tar → extracts locally → prints tail
3. A `CronCreate` job fires every 2-5 minutes calling the fetch steps
4. Each cron tick surfaces the latest metrics and logs, enabling mid-training corrections

```bash
# Cron prompt template:
# 1. Check session alive: colab sessions | grep <session>
# 2. Tar output on VM: echo '... subprocess.run(["tar",...]) ...' | colab exec -s <name> --timeout 15
# 3. Download: colab download -s <name> /content/<output>.tar.gz <local>/output.tar.gz
# 4. Extract: tar -xzf output.tar.gz
# 5. Report: tail log + tail CSV + ls pngs/
```

**Why this matters:**
- **Live monitoring without WebSocket:** Download goes through REST, which stays up even when exec is dead
- **Early correction:** See plateauing/divergence early — fix hyperparameters and re-launch before the session expires
- **Incremental sync:** Each fetch pulls the latest PNGs and metrics, giving a real-time dashboard of training health
- **Session-aware:** If the session died between ticks, the cron catches it immediately (no wasted wait)

**When to use:** Any Colab training run expected to last >5 minutes. Cancel the cron when training completes; the fetched artifacts remain as a permanent record.

**Caveat:** `colab exec` (used for tar) can still fail via WebSocket. If exec fails, fall back to `colab download` directly on known output paths — the training process may have written new files even if exec is unreachable.

## High-signal training outputs (AI/ML/RL)

Every training script must produce three artifact types. The goal is **glance-and-decide**: one look at the fetched outputs tells you whether to let it run, fix and re-launch, or stop.

### 1. Logs (`logs/train.log`)

Timestamped per-N-episodes, not per-step. Each line must be self-contained — copy-paste-able into a comparison.

```
[HH:MM:SS] Ep 1200 | reward=203 | avg100=153.6 | eps=0.165 | q_mean=0.347 | elapsed=17s
```

Rules:
- Log at fixed intervals (every 20-50 episodes), plus first and last episode
- Include the key metric + its moving average (avg100) — raw reward alone is noise
- Include a convergence indicator (epsilon, learning rate, TD error) to distinguish "still exploring" from "plateaued"
- Include elapsed time — catch slowdowns early (memory leak, CPU bottleneck)
- Log evaluation runs separately with greedy/noiseless policy

### 2. Figures (`pngs/training_curves.png`)

One multi-panel figure, overwritten every N episodes (100-200). This is the visual dashboard.

Minimum panels for RL:
- **Reward curve** (raw + moving average + solved threshold line)
- **Episode length** (steps survived)
- **Exploration decay** (epsilon over time)
- **Value distribution** (Q-value histogram, loss curve, or entropy)

Rules:
- Always include a horizontal reference line (e.g., "solved = 195") — naked curves are ambiguous
- Overwrite, don't accumulate files — `fetch.sh` always gets the latest snapshot
- Use `matplotlib.use("Agg")` for headless VMs

### 3. Metrics CSV (`metrics.csv`)

One row per episode, all scalar metrics as columns. This is the raw data for offline analysis.

```
episode,reward,steps,epsilon,avg100_reward,q_mean,q_max,elapsed_s,td_error_mean
```

Rules:
- Write header on creation, append each episode — crash-safe (no in-memory accumulation)
- Include everything needed to reproduce a plot from the CSV alone
- Final row should be the last completed episode — never missing data

### Why this matters

| Without structured outputs | With structured outputs |
|---------------------------|------------------------|
| "Is it learning? Not sure, let me wait longer" | avg100 climbed 22→148 in 5 min, plateauing — fix epsilon decay now |
| 30 min wasted on a diverged run | Caught at minute 2 from Q-value histogram blowing up |
| Can't compare v1 vs v2 | `diff metrics.csv` shows exactly where v2 diverged |
| Session dies, all evidence lost | Last cron fetch preserved everything up to ep 2500 |

**Bottom line:** The cron watchtower fetches artifacts. Artifact quality determines whether the watchtower sees anything useful. Design outputs so that a 5-second glance at the fetched tail tells you the next action.

## Deploy scripts (bash)

**macOS default bash is 3.2.** No associative arrays (`declare -A`), no `shopt -s globstar`. Use indexed arrays only.

**Aliases don't work in bash scripts.** `cb`, `clb`, `cc` are zsh aliases — not available in `#!/bin/bash`. Always use explicit env vars:
```bash
HOME=~/colab-accounts/account-b /Users/mx/.local/bin/colab new --gpu T4 -s <name>
HOME=~/colab-accounts/account-clb /Users/mx/.local/bin/colab exec -s <name> -f script.py
```

**Never expand proxy vars via `$VAR`.** `env $PROXY_VARS` concatenates multi-value strings, causing `InvalidSchema`/`LocationParseError`. Always use explicit per-variable `export` lines:
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

## HF datasets: avoid the library, use raw CDN

Colab's pre-installed `datasets` version is too new for many dataset scripts. Download raw files directly from HF CDN:

```python
# Pattern: https://huggingface.co/datasets/ORG/REPO/resolve/main/path/to/file
# Use canonical org casing — lowercase redirects via 307 (urllib doesn't follow)
url = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/de/en/de-en.zip"
urllib.request.urlretrieve(url, dest)  # 302 redirect — works
```

`urllib.request.urlretrieve` follows 301/302/303 but NOT 307/308. HF CDN uses 307 for case mismatches, 302 for canonical URLs. Always use canonical org name casing.

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

## Pre-deploy checklist

- Run forward pass locally (random tensor) to verify model output shape and no NaN
- Fit PCA on a sample locally to verify resize → stack doesn't crash
- Validate data pipeline loads images correctly (check first batch shapes + labels)
- `grep` the codebase for `load_dataset` — if found, verify HF dataset name still works on Colab's `datasets` version
- For Kaggle: verify `kernel-metadata.json` has `enable_gpu: true`, `enable_internet: true`, correct `id` slug, and `kernel_type: "script"`

## Doc protocol

- After writing docs/charts, open them for review. Don't claim "done" without visual verification.
- Write gotchas.md per-project proactively — don't wait to be asked.

### Gotcha triage: where to save what

When a problem, surprise, or workaround is discovered during a session, route it by scope:

| Scope | Destination | Example |
|-------|------------|---------|
| Specific to one project | `projects/<project>/gotchas.md` | "PCA resize must use fixed [H,W], not single int" (alexnet) |
| Colab/Kaggle CLI mechanics | `.claude/skills/<skill>/references/gotchas.md` | "`colab upload` creates file not dir when path missing" (colab-cli) |
| ML/model-wide, not CLI-specific | `docs/model-gotchas.md` | "Beam search eval OOMs even when training fits — allocate extra cache" |

**Decision rule:** If the learning is about *how to use the tool* (colab exec, upload, proxy, sessions), it goes to the skill gotchas. If it's about *the model/algorithm behavior* (init scheme, CUDA OOM, convergence), it goes to the project gotchas. If it applies across projects (any transformer, any RL agent), it goes to `docs/model-gotchas.md`. When uncertain, default to the project gotchas — it's easier to promote later than to find a buried note.

## Multi-session awareness

- User runs parallel Claude Code sessions (41% of messages). Files may change mid-session.
- Skills live in `.claude/skills/<name>/SKILL.md` — not `.agents/skills/` or other paths.
