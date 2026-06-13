# colab-cli

**GPU training from the terminal.** Provision Colab/Kaggle VMs, execute training scripts, monitor progress, and fetch artifacts — all without opening a browser.

<p align="center">
  <a href="https://1998x-stack.github.io/colab-cli/"><strong>🌐 Landing Page</strong></a> &nbsp;·&nbsp;
  <a href="#-projects"><strong>Projects</strong></a> &nbsp;·&nbsp;
  <a href="#-benchmark"><strong>Benchmark</strong></a> &nbsp;·&nbsp;
  <a href="#-quickstart"><strong>Quickstart</strong></a> &nbsp;·&nbsp;
  <a href="./projects/REPORT_ddpg_vs_td3.md"><strong>DDPG vs TD3 Report</strong></a>
</p>

---

## Features

- **One-command GPU provisioning** — `colab new --gpu T4` gets you a T4 VM in seconds
- **Detached background training** — `start_new_session=True` survives WebSocket drops behind proxies
- **Cron-based monitoring** — fetch logs, metrics, and plots every N minutes via `fetch.sh`
- **Multi-account parallelism** — 4 Colab + 4 Kaggle accounts for parallel GPU jobs
- **China proxy resilience** — SOCKS5/HTTP CONNECT flip pattern for GFW-bypassed REST + WebSocket
- **Full artifact pipeline** — train on VM → save checkpoints → download plots/metrics → local analysis

## Quickstart

```bash
# 1. Install
uv tool install google-colab-cli

# 2. Authenticate
colab --auth oauth2 new

# 3. Provision GPU
colab new --gpu T4 -s training

# 4. Upload & launch (detached, survives proxy drops)
colab upload train.py /content/train.py
colab exec -f launch.py --timeout 120

# 5. Monitor — fetch artifacts on a cron
./fetch.sh                          # downloads train.log, metrics.json, progress.png

# 6. Tear down
colab stop -s training
```

> **From China?** See [Proxy Setup](#-proxy-setup) below. The two-config flip handles GFW interference.

## Projects

A growing collection of ML experiments trained on free-tier Colab/Kaggle GPUs.

### Reinforcement Learning

| Project | Algorithm | Environment | GPU | Peak Result |
|---------|-----------|-------------|-----|-------------|
| [`td3-gym`](./projects/td3-gym/) | **TD3** | Pendulum-v1 | T4 | **-71.76** (final = best) |
| [`ddpg-gym`](./projects/ddpg-gym/) | **DDPG** | Pendulum-v1 | T4 | -40.95 (lost after ep 40) |
| [`rl-sac`](./projects/rl-sac/) | SAC | MountainCarContinuous-v0 | T4 | — |
| [`rl-dqn-atari`](./projects/rl-dqn-atari/) | DQN | Atari | T4 | — |

### Computer Vision

| Project | Model | Dataset | GPU |
|---------|-------|---------|-----|
| [`alexnet_imagenette`](./projects/alexnet_imagenette/) | AlexNet (faithful repro) | Imagenette (10-class) | T4 |
| [`vit-cifar10`](./projects/vit-cifar10/) | Vision Transformer | CIFAR-10 | Kaggle P100 |
| [`cnn-cifar10`](./projects/cnn-cifar10/) | CNN from scratch | CIFAR-10 | T4 |

### NLP & LLMs

| Project | Task | GPU |
|---------|------|-----|
| [`transformer_iwslt`](./projects/transformer_iwslt/) | Attention Is All You Need — IWSLT'14 De→En | T4 |
| [`vllm-compare`](./projects/vllm-compare/) | vLLM inference benchmarks (SmolLM2, Qwen2.5) | T4 |
| [`vllm-rag`](./projects/vllm-rag/) | RAG pipeline with vLLM | T4 |
| [`hotpotqa-reasoning`](./projects/hotpotqa-reasoning/) | CoT vs ReAct prompting | T4 |
| [`nanogpt`](./projects/nanogpt/) | NanoGPT training experiments | T4 |
| [`nanochat-colab`](./projects/nanochat-colab/) | NanoChat on Colab | T4 |
| [`rnn-imdb`](./projects/rnn-imdb/) | RNN sentiment analysis | T4 |

### Tutorials & Infrastructure

| Project | Description |
|---------|-------------|
| [`cuda-tutorial`](./projects/cuda-tutorial/) | CUDA kernel tutorials |
| [`ml-tutorial`](./projects/ml-tutorial/) | ML fundamentals |
| [`autoresearch-t4`](./projects/autoresearch-t4/) | Automated ML research on T4 |

## Benchmark: DDPG vs TD3

We ran identical Pendulum-v1 training (200 episodes, seed 42, T4 GPU) to compare stability.

| Metric | DDPG | TD3 |
|--------|------|-----|
| **Best eval** | -40.95 (ep 40) | **-71.76** (ep 200) |
| **Final eval** | -165.96 ± 56.8 | **-71.76** ± 93.2 |
| **Best = final?** | No — lost at ep 50 | **Yes** |
| **Catastrophic forgetting** | Yes (-606 drop) | **None** (0.0) |
| **Last-5 eval stdev** | 51.1 | **27.9** |
| **Eval trend** | Erratic, plateaued | Monotonic, still improving |

**Key finding**: DDPG found a higher peak (-40.95) but immediately lost it. TD3 converged more slowly but its final model is its best model — the policy was still improving at episode 200. For deployment (where you ship the final checkpoint, not a mid-training snapshot), TD3 wins decisively.

**Full report**: [`projects/REPORT_ddpg_vs_td3.md`](./projects/REPORT_ddpg_vs_td3.md) — 20-point eval table, block-average trends, stability analysis, interactive charts on the [landing page](https://1998x-stack.github.io/colab-cli/).

## Proxy Setup

Colab APIs are blocked in mainland China. Two network paths need different proxy treatment.

```bash
# Config A — SOCKS5 for REST, WebSocket direct (try first):
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

# Config B — HTTP CONNECT tunnel (flip if A returns 503):
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

**Diagnosing GPU failures**:

| Config | Error on `colab new --gpu T4` | Action |
|--------|-------------------------------|--------|
| A (SOCKS5 + no_proxy) | `Service Unavailable` (503) | Flip to config B |
| B (HTTP CONNECT) | `Precondition Failed` (412) | GPU exhausted — try other accounts |
| All accounts | 412 | Use CPU fallback: omit `--gpu` |

Config B handles the full workflow (new, upload, exec, download) without switching.

## Accounts

Four Colab accounts via isolated `$HOME` directories, four Kaggle accounts via token files. Only 1 GPU per free account — multi-account enables parallel training.

```bash
colab   # hackxie1998 (default)    cb    # stefaniehu929
cc      # xbetterdetermine          clb   # xieminghack
```

```bash
# Kaggle: swap tokens per kernel
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./project
```

See [`.claude/skills/colab-cli/SKILL.md`](./.claude/skills/colab-cli/SKILL.md) and [`.claude/skills/kaggle-cli/SKILL.md`](./.claude/skills/kaggle-cli/SKILL.md) for full multi-account workflows.

## Repository Structure

```
.claude/skills/           # Claude Code skills (colab-cli, kaggle-cli)
├── colab-cli/            # Full Colab CLI workflow + 22 gotchas
└── kaggle-cli/           # Full Kaggle CLI workflow + 16 gotchas
docs/                    # GitHub Pages landing page + Chinese guide
projects/                # ML training projects (17 total)
├── ddpg-gym/            # DDPG on Pendulum (train.py, launch.py, fetch.sh)
├── td3-gym/             # TD3 on Pendulum (train.py, launch.py, fetch.sh)
├── REPORT_ddpg_vs_td3.md # Full benchmark report
├── alexnet_imagenette/  # AlexNet faithful reproduction
├── transformer_iwslt/   # Transformer De→En translation
├── vit-cifar10/         # Vision Transformer
├── vllm-compare/        # vLLM model benchmarks
├── rl-sac/ rl-dqn-atari/ cnn-cifar10/ nanogpt/ nanochat-colab/ rnn-imdb/
└── vllm-rag/ cuda-tutorial/ ml-tutorial/ autoresearch-t4/ hotpotqa-reasoning/
```

## Key Gotchas

- **`colab exec -f` reads LOCAL files** (relative to CWD), not remote VM files. Upload is only needed for subprocess-spawned scripts.
- **First Colab session rarely produces useful training** — data download + CUDA JIT = 7-10 min overhead. Second session with cached data works normally.
- **stdout is buffered in subprocess** — set `PYTHONUNBUFFERED=1` and use `python -u` in background jobs.
- **Empty logs ≠ stuck job** — verify with `nvidia-smi` or check for side effects before killing.
- **`colab download` doesn't do directories** — tar first: `tar -czf out.tar.gz -C /content dir/`.
- **Checkpoints >600MB fail through proxy** — save weights-only for download.
- **SOCKS5 proxy needs PySocks** — `pip install requests[socks]` or use HTTP style.
- **macOS bash is 3.2** — no associative arrays, no `shopt -s globstar`. Use indexed arrays only.
- **Shell aliases don't work in bash scripts** — use explicit `HOME=~/colab-accounts/account-b /path/to/colab`.

## Docs

- [`docs/DeepSeek-Colab CLI 深度使用指南.md`](./docs/DeepSeek-Colab%20CLI%20深度使用指南.md) — Comprehensive guide (Chinese)
- [`docs/multi-account-colab.md`](./docs/multi-account-colab.md) — Multi-account setup
- [`.claude/skills/colab-cli/references/gotchas.md`](./.claude/skills/colab-cli/references/gotchas.md) — 22 field-tested gotchas
- [`.claude/skills/colab-cli/references/workflows.md`](./.claude/skills/colab-cli/references/workflows.md) — Full workflow patterns
- [`projects/REPORT_ddpg_vs_td3.md`](./projects/REPORT_ddpg_vs_td3.md) — DDPG vs TD3 benchmark

## License

MIT
