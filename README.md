# colab-cli

**GPU training from the terminal.** Provision Colab/Kaggle VMs, execute training scripts, monitor progress, and fetch artifacts — all without opening a browser.

<p align="center">
  <a href="https://1998x-stack.github.io/colab-cli/"><strong>🌐 Landing Page</strong></a> &nbsp;·&nbsp;
  <a href="#-projects"><strong>Projects</strong></a> &nbsp;·&nbsp;
  <a href="#-benchmark"><strong>Benchmark</strong></a> &nbsp;·&nbsp;
  <a href="#-quickstart"><strong>Quickstart</strong></a> &nbsp;·&nbsp;
  <a href="./projects/rl/REPORT_ddpg_vs_td3.md"><strong>DDPG vs TD3 Report</strong></a>
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

### Reinforcement Learning (`projects/rl/`)

| Project | Algorithm | Environment | GPU | Peak Result |
|---------|-----------|-------------|-----|-------------|
| [`td3-gym`](./projects/rl/td3-gym/) | **TD3** | Pendulum-v1 | T4 | **-71.76** (final = best) |
| [`ddpg-gym`](./projects/rl/ddpg-gym/) | **DDPG** | Pendulum-v1 | T4 | -40.95 (lost after ep 40) |
| [`ddpg-td3-mujoco`](./projects/rl/ddpg-td3-mujoco/) | DDPG + TD3 | MuJoCo (3 envs) | T4 | Head-to-head comparison |
| [`ddqn-noisy-ram`](./projects/rl/ddqn-noisy-ram/) | Double DQN + NoisyNet | Atari RAM | T4 | Prioritized ER |
| [`ppo-atari-ram`](./projects/rl/ppo-atari-ram/) | PPO + GAE | Atari RAM (63 envs) | T4 | Two-tier config system |
| [`ppo-mujoco`](./projects/rl/ppo-mujoco/) | PPO + GAE | MuJoCo | T4 | Gaussian policy |
| [`rl-dqn-atari`](./projects/rl/rl-dqn-atari/) | DQN + Dueling | ALE/Pong-v5 | T4 | CNN encoder |
| [`rl-sac`](./projects/rl/rl-sac/) | SAC + auto entropy | MountainCarContinuous-v0 | T4 | — |
| [`rl-sarsa-gym`](./projects/rl/rl-sarsa-gym/) | Tabular SARSA | CartPole-v1 | T4 | 12-bin discretization |

### Computer Vision (`projects/cv/`)

| Project | Model | Dataset | GPU |
|---------|-------|---------|-----|
| [`alexnet_imagenette`](./projects/cv/alexnet_imagenette/) | AlexNet (faithful repro) | Imagenette-160 | T4 |
| [`vit-cifar10`](./projects/cv/vit-cifar10/) | Vision Transformer (scratch) | CIFAR-10 | Kaggle P100 |
| [`cnn-cifar10`](./projects/cv/cnn-cifar10/) | 3-block CNN | CIFAR-10 | T4 |
| [`cnn-quantization`](./projects/cv/cnn-quantization/) | ResNet-18 FP32/FP16/INT8/INT4 | CIFAR-10 | T4 |
| [`cnn-explainer`](./projects/cv/cnn-explainer/) | CNN + XAI (Grad-CAM, Saliency) | CIFAR-10 | T4 |

### NLP & LLMs (`projects/nlp/`)

| Project | Task | GPU |
|---------|------|-----|
| [`transformer_iwslt`](./projects/nlp/transformer_iwslt/) | Transformer De→En translation | T4 |
| [`nanogpt`](./projects/nlp/nanogpt/) | NanoGPT char-level LM | T4 |
| [`nanochat-colab`](./projects/nlp/nanochat-colab/) | NanoChat full-stack ChatGPT | T4 |
| [`s1-t4`](./projects/nlp/s1-t4/) | Test-time scaling (QLoRA Qwen2.5-7B) | T4 |
| [`seq2seq-t4`](./projects/nlp/seq2seq-t4/) | Seq2Seq with attention | T4 |
| [`rnn-imdb`](./projects/nlp/rnn-imdb/) | BiLSTM sentiment analysis | T4 |
| [`word2vec-c4`](./projects/nlp/word2vec-c4/) | Skip-gram on C4 corpus | T4 |
| [`rag-fasttext`](./projects/nlp/rag-fasttext/) | Hybrid RAG: BM25 + FastText + FAISS | T4 |
| [`vllm-compare`](./projects/nlp/vllm-compare/) | vLLM inference benchmarks | T4 |
| [`vllm-rag`](./projects/nlp/vllm-rag/) | RAG pipeline with vLLM + ChromaDB | T4 |
| [`hotpotqa-reasoning`](./projects/nlp/hotpotqa-reasoning/) | CoT vs ReAct prompting | T4 |

### Graph Neural Networks (`projects/gnn/`)

| Project | Model | Datasets | GPU |
|---------|-------|----------|-----|
| [`gnn-citation`](./projects/gnn/gnn-citation/) | 2-layer GCN | Cora, CiteSeer, PubMed | T4 |

### Systems & Infrastructure (`projects/systems/`)

| Project | Description |
|---------|-------------|
| [`autoresearch-t4`](./projects/systems/autoresearch-t4/) | Autonomous LLM pretraining research (5-min budget) |
| [`cuda-tutorial`](./projects/systems/cuda-tutorial/) | 7 progressive CUDA kernel tutorials |
| [`swe-agent-colab`](./projects/systems/swe-agent-colab/) | SWE-agent on Colab with vLLM |

### Tutorials (`projects/tutorials/`)

| Project | Description |
|---------|-------------|
| [`ml-tutorial`](./projects/tutorials/ml-tutorial/) | NLP, CV, Audio — fine-tune pretrained transformers |

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

**Full report**: [`projects/rl/REPORT_ddpg_vs_td3.md`](./projects/rl/REPORT_ddpg_vs_td3.md) — 20-point eval table, block-average trends, stability analysis, interactive charts on the [landing page](https://1998x-stack.github.io/colab-cli/).

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
.claude/skills/                  # Claude Code skills (colab-cli, kaggle-cli)
├── colab-cli/                   # Full Colab CLI workflow + 22 gotchas
└── kaggle-cli/                  # Full Kaggle CLI workflow + 16 gotchas
docs/                            # Documentation
├── guides/                      # How-to guides (6 docs)
├── reference/                   # Technical deep-dives (5 docs)
├── superpowers/                 # Superpowers plans and specs
├── google-workspace-mcp/        # Google Workspace MCP integration
└── plots/                       # Generated plots
papers/                          # Research paper notes
├── s1/                          # s1: Simple test-time scaling
└── seq2seq/                     # Seq2Seq with attention
projects/                        # ML training projects (30 total, 6 categories)
├── rl/                          # Reinforcement Learning (9 projects)
│   ├── td3-gym/ ddpg-gym/ ddpg-td3-mujoco/ ddqn-noisy-ram/
│   ├── ppo-atari-ram/ ppo-mujoco/ rl-dqn-atari/ rl-sac/ rl-sarsa-gym/
│   └── REPORT_ddpg_vs_td3.md
├── cv/                          # Computer Vision (5 projects)
│   ├── alexnet_imagenette/ cnn-cifar10/ cnn-explainer/
│   └── cnn-quantization/ vit-cifar10/
├── nlp/                         # NLP & LLMs (11 projects)
│   ├── transformer_iwslt/ nanogpt/ nanochat-colab/ s1-t4/ seq2seq-t4/
│   ├── rnn-imdb/ word2vec-c4/ rag-fasttext/ hotpotqa-reasoning/
│   └── vllm-compare/ vllm-rag/
├── gnn/                         # Graph Neural Networks (1 project)
│   └── gnn-citation/
├── systems/                     # Systems & Infrastructure (3 projects)
│   └── autoresearch-t4/ cuda-tutorial/ swe-agent-colab/
└── tutorials/                   # Education (1 project)
    └── ml-tutorial/
index.md                         # Full project index with descriptions
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

- [`docs/guides/`](./docs/guides/) — How-to guides (Colab CLI, multi-account, quantization, session monitoring)
- [`docs/reference/`](./docs/reference/) — Technical deep-dives (model gotchas, WebSocket analysis, Kaggle analysis)
- [`docs/superpowers/`](./docs/superpowers/) — Superpowers plans and specs
- [`.claude/skills/colab-cli/references/gotchas.md`](./.claude/skills/colab-cli/references/gotchas.md) — 22 field-tested gotchas
- [`.claude/skills/colab-cli/references/workflows.md`](./.claude/skills/colab-cli/references/workflows.md) — Full workflow patterns
- [`projects/rl/REPORT_ddpg_vs_td3.md`](./projects/rl/REPORT_ddpg_vs_td3.md) — DDPG vs TD3 benchmark
- [`index.md`](./index.md) — Full project index with descriptions

## License

MIT
