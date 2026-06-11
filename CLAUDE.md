# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
    └── scripts/             # push_and_wait.py, check_progress.py

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

## Accounts

**Colab (4 accounts)** — isolated via `$HOME` directories. See colab-cli skill for full details.

```bash
colab   # hackxie1998 (default)    cb    # stefaniehu929
cc      # xbetterdetermine          clb   # xieminghack
```

**Kaggle (4 accounts)** — tokens in `.kaggle/access_token{1,2,3,4}`. Switch via env var per command:

```bash
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./project  # xieming1998
KAGGLE_API_TOKEN="$(cat .kaggle/access_token1)" kaggle kernels push -p ./project  # account 1
```

See kaggle-cli skill for multi-account management, kernel workflows, and GPU compatibility.

## Proxy (Colab only)

Colab uses **two separate network paths** (see `docs/websocket-stability-analysis.md`):

- **REST API** (`colab.pa.googleapis.com`): `colab new`, `colab stop`, `colab sessions`, `colab download`, keep-alive. Uses `requests` — auto-detects `HTTP_PROXY`/`HTTPS_PROXY`.
- **WebSocket** (`*.prod.colab.dev`): `colab exec`, `colab upload`, `colab ls`. Uses `websocket-client` — does NOT pass proxy params, can't parse `socks5://` from env vars.

**Critical:** `colab upload` goes through WebSocket — it will fail with 500 errors from China when the exec WebSocket is unstable. For multi-file projects, use the base64 embed pattern: generate a Python script that writes all files to `/content/` and exec that single script. See colab-cli gotchas.md §"Multi-file deploy: use base64 embed, not upload".

**Recommended config** — REST through SOCKS5 proxy, WebSocket direct:

```bash
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

If WebSocket direct fails, try without `no_proxy` (WebSocket through proxy — slower, may work). Flip per session.

```bash
# colab (hackxie1998) — default account
colab new --gpu T4 -s <name> && colab exec -s <name> -f script.py --timeout 120

# cc (xbetterdetermine), cb (stefaniehu929), clb (xieminghack) — same pattern
HOME=~/colab-accounts/account-c colab new --gpu T4 -s <name>
HOME=~/colab-accounts/account-c colab exec -s <name> -f script.py --timeout 120
```

Only 1 GPU per free account. SSL errors transient — retry before assuming dead.

Colab official limits: 12h max session, ~90min idle timeout. The observed ~12-15min effective window is WebSocket disconnection through the proxy, NOT Colab killing the session. The keep-alive daemon (auto-spawned by `colab new`, calls `KeepAliveAssignment` RPC every 60s) prevents idle timeout — but does nothing for exec WebSocket stability.

## Free-tier reality

**Colab official limits**: 12h max session, ~90min idle timeout. GPU quota is dynamic — heavy use triggers 12-24h cooldown.

**Effective reality from China**: `colab exec` WebSocket drops frequently through the proxy, making interactive work windows ~12-15 min. The session itself survives (keep-alive daemon works), but exec becomes unreachable.

**Mitigations**:
- Detached bootstrap (`launch.py` spawns training via `start_new_session=True`) — training survives exec drops
- `no_proxy="*.colab.dev"` — WebSocket direct often more stable than through SOCKS5
- Checkpoint to Drive (`colab drivemount`) — survive session death
- Multi-account rotation — work around GPU quota cooldowns
- Kaggle Notebooks as complement — push model (REST API), no WebSocket dependency, 30h/week GPU

See `docs/` for deep-dive analysis: websocket-stability, session-health-monitoring, kaggle-notebooks, drive-mcp-colab.

## Core workflow

**Colab (interactive, WebSocket):**
```bash
# 1. Provision + mount Drive + upload + launch
colab new --gpu T4 -s training
colab drivemount -s training
colab upload *.py /content/
colab exec -s training -f launch.py --timeout 120   # detaches train subprocess

# 2. Monitor (local cron + VM watchdog)
CronCreate cron="*/5 * * * *" prompt="Check session..." durable=true recurring=true

# 3. Download + cleanup
colab download /content/<project>-output.tar.gz projects/<project>/output/
colab stop -s training
```

**Kaggle (push, REST only):**
```bash
# 1. Configure metadata + push (single REST call, returns immediately)
cat > kernel-metadata.json << EOF
{"id":"xieming1998/my-training","title":"My Training","code_file":"train.py",
 "language":"python","kernel_type":"script","is_private":true,
 "enable_gpu":true,"enable_internet":true,...}
EOF
kaggle kernels push -p .

# 2. Monitor (REST polling, no connection to maintain)
kaggle kernels status xieming1998/my-training
kaggle kernels logs xieming1998/my-training

# 3. Download output
kaggle kernels output xieming1998/my-training -p ./output
```

## Checkpoint persistence

**VM-local checkpoints (`/content/checkpoints/`) die with the session.** Two strategies:

1. **Drive mount (P0, recommended)**: `colab drivemount` → train.py writes checkpoints to `/content/drive/MyDrive/colab-checkpoints/<project>/`. VM→Drive goes over Google internal network — bypasses China proxy entirely.
2. **Manual tar+download**: Cron-triggered download of checkpoint tars. Works but adds complexity.

See `docs/drive-mcp-colab-integration.md` for full analysis.

## Detached training (gotchas)

- `PYTHONUNBUFFERED=1` + `python -u` + `start_new_session=True` in subprocess.Popen
- `colab exec -f` reads LOCAL files (relative paths), sends to VM. No `-c` flag — use stdin pipe.
- `colab download` needs tar for dirs: `tar -czf /content/out.tar.gz -C /content dir/`
- **Log buffering despite PYTHONUNBUFFERED**: stdout to file via Popen can still buffer. Training appears stuck but GPU is active. Add `flush=True` to all print() calls. Add per-N-batch progress prints. Verify with `nvidia-smi`.
- **Checkpoint downloads >600MB fail through proxy**: Full checkpoint with optimizer state = ~1GB, proxy connection breaks at ~624MB (IncompleteRead). Save a separate **weights-only checkpoint** (~120-233MB) for download, keep full checkpoint for VM-local resume.
- **BLEU/beam search is the hidden bottleneck**: Beam search eval on full val set takes hours. Use 100-sentence subset with greedy decode for training-time eval.
- **First session wastes a VM**: Data prep + CUDA JIT = 7-10 min overhead. First epoch rarely completes. Second session (data cached) gets 3-4 epochs. Account for this in session budget.
- **CUDA OOM during eval even when training fits**: Beam search allocates extra tensors. Use `torch.cuda.empty_cache()` before eval. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Deploy scripts (bash)

**macOS default bash is 3.2.** No associative arrays (`declare -A`), no `shopt -s globstar`. Use indexed arrays only.

**Aliases don't work in bash scripts.** `cb`, `clb`, `cc` are zsh aliases — not available in `#!/bin/bash` or non-interactive shells. Always use explicit env vars:
```bash
# Each account:
HOME=~/colab-accounts/account-b /Users/mx/.local/bin/colab new --gpu T4 -s <name>
HOME=~/colab-accounts/account-clb /Users/mx/.local/bin/colab exec -s <name> -f script.py
```

**Never expand proxy vars via `$VAR`.** `export $PX` and `env $PROXY` mangle multi-value env vars (URL parsing breaks with `InvalidSchema`/`LocationParseError`). Always use separate `export` lines:
```bash
# Works:
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
cmd ...

# Broken — URL concatenation:
export PX="HTTPS_PROXY=http://... HTTP_PROXY=http://..."
env $PX cmd        # env var values bleed together
export $PX && cmd  # same issue
```

## Proxy gotchas

**SOCKS5 needs PySocks for REST (colab download/sessions).** `socks5://127.0.0.1:7890` fails with `InvalidSchema: Missing dependencies for SOCKS support` on `colab download`. Install: `pip install requests[socks]`. Or use `http://127.0.0.1:7890` style for REST — works without extra deps.

**REST survives when WebSocket dies.** `colab sessions`, `colab download`, `colab stop` use REST (`colab.pa.googleapis.com`) — they work even when `colab exec` returns 404/401. Use `colab download` as a fallback monitoring path to check training progress when exec is unreachable.

## HF datasets: avoid the library, use raw CDN

**Colab's pre-installed `datasets` version is too new** for many dataset scripts. `load_dataset("iwslt2017", ...)` fails with "Dataset scripts are no longer supported." Pinning `datasets==2.14.0` partly works but lacks `trust_remote_code`.

**Instead, download raw files directly from HF CDN:**
```python
# Pattern: https://huggingface.co/datasets/ORG/REPO/resolve/main/path/to/file
# Use canonical org casing — lowercase redirects via 307 (urllib doesn't follow)
url = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/de/en/de-en.zip"
urllib.request.urlretrieve(url, dest)  # 302 redirect — works
```

**urllib vs redirects:** `urllib.request.urlretrieve` follows 301/302/303 but NOT 307/308. HF CDN uses 307 for case mismatches, 302 for canonical URLs. Always use the canonical org name casing.

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
- For Kaggle: verify `kernel-metadata.json` has `enable_gpu: true`, `enable_internet: true`, correct `id` slug, and `kernel_type: "script"`

## Doc protocol

- After writing docs/charts, open them for review. Don't claim "done" without visual verification.
- Write gotchas.md per-project proactively — don't wait to be asked.

## Multi-session awareness

- User runs parallel Claude Code sessions (41% of messages). Files may change mid-session.
- Skills live in `.claude/skills/<name>/SKILL.md` — not `.agents/skills/` or other paths.

## Kaggle Notebooks (complementary GPU)

Kaggle's push model (REST API) avoids Colab's WebSocket problem entirely. 30h/week GPU (P100 or T4 x2), transparent quota.

**The kaggle-cli skill (`.claude/skills/kaggle-cli/SKILL.md`) has the full workflow** — push, monitor, download, multi-account, GPU compatibility. Key commands:

```bash
kaggle kernels push -p ./project-dir     # push + run (REST call, no long connection)
kaggle kernels status user/slug          # check status
kaggle kernels logs user/slug            # view stdout/stderr
kaggle kernels output user/slug -p ./    # download results
```

**Four accounts** configured via `.kaggle/access_token{1,2,3,4}`. Switch per-command:
```bash
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./project
```

**Critical GPU gotcha:** P100 (sm_60) is incompatible with pre-installed PyTorch 2.10.0+cu128. Force-reinstall with CUDA 12.6 at script start. See skill references/gotchas.md.

See `docs/kaggle-notebooks-analysis.md` for full comparison and integration strategy.
