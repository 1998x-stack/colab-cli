# colab-cli

**GPU training from the terminal.** Provision Colab/Kaggle VMs, execute
training scripts, monitor progress, and fetch artifacts — all without opening
a browser. Built for reliability from mainland China behind the GFW.

<p align="center">
  <a href="https://1998x-stack.github.io/colab-cli/"><strong>Landing Page</strong></a> &nbsp;·&nbsp;
  <a href="#-root-cause-analysis"><strong>Root Cause Analysis</strong></a> &nbsp;·&nbsp;
  <a href="#-projects"><strong>Projects</strong></a> &nbsp;·&nbsp;
  <a href="#-quickstart"><strong>Quickstart</strong></a> &nbsp;·&nbsp;
  <a href="#-docs"><strong>Docs</strong></a>
</p>

---

## Root Cause Analysis

Free-tier Colab GPU sessions die after ~10 minutes — even with active training.
We traced this to its source and built a verified fix.

**The keep-alive daemon is broken.** Every `colab new --gpu T4` spawns a
background daemon that calls the `KeepAliveAssignment` RPC to tell Colab
"this session is still in use." But an IAM deadlock in the `x-goog-user-project`
header causes **HTTP 403 `USER_PROJECT_DENIED`** on every call. The daemon
exits 61 seconds after every session starts. Zero keep-alive pings ever reach
Colab's backend.

**The WebSocket connection is the real liveness signal.** We proved through
3 live T4 GPU experiments that the `colab exec` WebSocket connection — not the
keep-alive daemon — prevents GPU reclamation. While the WebSocket stays open,
the session stays alive. The session dies ~2-3 minutes after the last WebSocket
closes.

**Relay handoff extends sessions beyond the 10-minute cliff.** From China,
WebSocket connections drop at 8-12 minutes (carrier NAT + GFW). By chaining
multiple `colab exec` watchdogs with 7-minute windows and 1-minute overlap,
sessions achieve continuous coverage. The Jupyter kernel's serial execution
queue ensures handoff gaps of under 5 seconds — well within the ~2-minute
grace period.

| Document | Content |
|----------|---------|
| [`docs/colab-gpu-keepalive.md`](./docs/colab-gpu-keepalive.md) | Root cause: IAM deadlock, WebSocket liveness, relay handoff protocol |
| [`docs/websocket-stability-china.md`](./docs/websocket-stability-china.md) | China WebSocket drops: NAT/GFW/proxy layer analysis, ping gap |
| [`docs/core-flows.md`](./docs/core-flows.md) | Command-level sequence diagrams for all `colab` operations |
| [`docs/google-colab-cli-source-analysis.md`](./docs/google-colab-cli-source-analysis.md) | Full source architecture (google-colab-cli v0.5.11) |

## Features

- **One-command GPU provisioning** — `colab new --gpu T4` gets a T4 VM in seconds
- **Relay handoff for long training** — chain `colab exec` watchdogs to bypass the 10-min GPU timeout
- **WebSocket liveness monitoring** — keep sessions alive via kernel WebSocket, not the broken REST daemon
- **Cron-based artifact fetching** — download logs, metrics, and plots every N minutes via `colab download` (REST, survives WebSocket drops)
- **Multi-account parallelism** — 4 Colab + 4 Kaggle accounts for parallel GPU jobs
- **China proxy resilience** — two-config SOCKS5/HTTP CONNECT flip pattern, carrier NAT timeout analysis, GFW bypass
- **Full artifact pipeline** — structured training outputs (timestamped logs, metrics CSV, multi-panel PNGs, checkpoints)

## Quickstart

```bash
# 1. Install
uv tool install google-colab-cli

# 2. Authenticate
colab --auth oauth2 new

# 3. Provision GPU
colab new --gpu T4 -s training

# 4. Upload & launch (detached bootstrap, survives WebSocket drops)
colab upload train.py /content/train.py
colab exec -f launch.py --timeout 120

# 5. Monitor — fetch artifacts on a cron (REST-based, WS-independent)
./fetch.sh

# 6. Tear down
colab stop -s training
```

> **From China?** See [Proxy Setup](#-proxy-setup). Use Config B (HTTP CONNECT) by default. For training >8 min, use [relay handoff](./docs/colab-gpu-keepalive.md).
> **Don't have a GPU?** Use Kaggle (30h/week free P100): `kaggle kernels push -p ./project`. See [kaggle-cli skill](./.claude/skills/kaggle-cli/SKILL.md).

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
| [`text2sql_finetune`](./projects/nlp/text2sql_finetune/) | Qwen3-0.6B LoRA for Text-to-SQL | T4 |

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

Identical Pendulum-v1 training (200 episodes, seed 42, T4 GPU).

| Metric | DDPG | TD3 |
|--------|------|-----|
| **Best eval** | -40.95 (ep 40) | **-71.76** (ep 200) |
| **Final eval** | -165.96 ± 56.8 | **-71.76** ± 93.2 |
| **Best = final?** | No — lost at ep 50 | **Yes** |
| **Catastrophic forgetting** | Yes (-606 drop) | **None** (0.0) |
| **Last-5 eval stdev** | 51.1 | **27.9** |
| **Eval trend** | Erratic, plateaued | Monotonic, still improving |

DDPG found a higher peak (-40.95) but immediately lost it. TD3 converged
more slowly but its final model is its best model. For deployment, TD3 wins.

**Full report**: [`projects/rl/REPORT_ddpg_vs_td3.md`](./projects/rl/REPORT_ddpg_vs_td3.md)

## Proxy Setup

Colab APIs are blocked in mainland China. REST and WebSocket use different
network paths and need different proxy treatment.

```bash
# Config B — HTTP CONNECT tunnel (RECOMMENDED, start here):
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# Config A — SOCKS5 for REST, WebSocket direct (flip if B fails):
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

**Diagnosing GPU provisioning failures:**

| Config | Error on `colab new --gpu T4` | Meaning |
|--------|-------------------------------|---------|
| A (SOCKS5 + no_proxy) | `Service Unavailable` (503) | Proxy issue OR GPU exhaustion — ambiguous |
| B (HTTP CONNECT) | `Precondition Failed` (412) | Genuine GPU quota exhaustion |

If Config B returns 412, try other accounts (`cb`, `cc`, `clb`). If ALL
accounts return 412, GPU is globally unavailable — use CPU fallback (omit
`--gpu`) or Kaggle.

**Key insight** (`docs/websocket-stability-china.md`): WebSocket connections
from China drop after 8-12 minutes due to carrier NAT timeouts (Telecom 8-10
min, Unicom 10-12 min, Mobile 10-15 min). WebSocket ping frames (opcode 0x9)
are 2-byte control frames with no application payload — most carrier NAT
devices don't count them as activity. The `reconnect_interval=0` in
`jupyter_kernel_client` means zero automatic recovery.

## Accounts

Four Colab accounts via isolated `$HOME` directories, four Kaggle accounts
via token files. Only 1 GPU per free account — multi-account enables parallel
training.

```bash
colab   # hackxie1998 (default)    cb    # stefaniehu929
cc      # xbetterdetermine          clb   # xieminghack
```

```bash
# Kaggle: swap tokens per kernel
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./project
```

See [`.claude/skills/colab-cli/SKILL.md`](./.claude/skills/colab-cli/SKILL.md)
and [`.claude/skills/kaggle-cli/SKILL.md`](./.claude/skills/kaggle-cli/SKILL.md)
for full multi-account workflows.

## Repository Structure

```
.claude/skills/                  # Claude Code skills (colab-cli, kaggle-cli)
├── colab-cli/                   # Full Colab CLI workflow + gotchas
└── kaggle-cli/                  # Full Kaggle CLI workflow + gotchas
docs/                            # Documentation
├── colab-gpu-keepalive.md       # Root cause: IAM deadlock, WebSocket liveness, relay
├── websocket-stability-china.md # China WS drops: NAT/GFW/proxy layer analysis
├── core-flows.md                # Command-level sequence diagrams
├── google-colab-cli-source-analysis.md  # Full source architecture (v0.5.11)
├── guides/                      # How-to guides (6 docs)
├── reference/                   # Technical deep-dives (8 docs)
└── google-workspace-mcp/        # Google Workspace MCP integration
papers/                          # Research paper notes
├── s1/                          # s1: Simple test-time scaling
└── seq2seq/                     # Seq2Seq with attention
projects/                        # ML training projects (30 total, 6 categories)
├── rl/                          # Reinforcement Learning (9 projects)
├── cv/                          # Computer Vision (5 projects)
├── nlp/                         # NLP & LLMs (11 projects)
├── gnn/                         # Graph Neural Networks (1 project)
├── systems/                     # Systems & Infrastructure (3 projects)
└── tutorials/                   # Education (1 project)
tests/ws-keepalive/              # WebSocket keepalive test scripts + output
index.md                         # Full project index with descriptions
```

## Key Gotchas

- **Keep-alive daemon is broken.** The `KeepAliveAssignment` RPC fails with
  HTTP 403 on every session (IAM deadlock). Sessions rely on WebSocket
  liveness instead. See [`docs/colab-gpu-keepalive.md`](./docs/colab-gpu-keepalive.md).
- **WebSocket drops at 8-12 min from China.** Carrier NAT doesn't recognize
  WebSocket ping frames as activity. Use relay handoff for training >8 min.
- **`colab exec -f` reads LOCAL files** (relative to CWD), not remote VM files.
  Upload is only needed for subprocess-spawned scripts.
- **`colab upload` goes to `*.prod.colab.dev` via REST** (HTTPS PUT), not
  WebSocket. It survives WebSocket drops. Same domain, different transport.
- **First Colab session rarely completes training** — data download + CUDA JIT
  = 7-10 min overhead. Use a warmup session first.
- **stdout is buffered in subprocess.** Set `PYTHONUNBUFFERED=1` and use
  `python -u` in background jobs, plus `flush=True` on all `print()` calls.
- **Empty logs ≠ stuck job.** Verify with `nvidia-smi` or check for side effects
  (files appearing, GPU utilization) before assuming the job is dead.
- **Inline Python via stdin is unreliable.** `echo '...' | colab exec` can
  fail silently for multi-line scripts. Always use `-f <file.py>` for
  production watchdogs and launchers.
- **`colab download` doesn't do directories.** Tar on VM first:
  `tar -czf out.tar.gz -C /content dir/`.
- **Checkpoints >600MB fail through proxy.** Save weights-only checkpoint
  (~120-233MB) for download; keep full optimizer state on VM.
- **macOS bash is 3.2.** No associative arrays, no `shopt -s globstar`.
- **Shell aliases don't work in bash scripts.** Use explicit
  `HOME=~/colab-accounts/account-b /path/to/colab` in `#!/bin/bash` scripts.

## Docs

- [`docs/colab-gpu-keepalive.md`](./docs/colab-gpu-keepalive.md) — Root cause: IAM deadlock, WebSocket liveness, relay handoff protocol
- [`docs/websocket-stability-china.md`](./docs/websocket-stability-china.md) — China WebSocket drops: NAT/GFW/proxy layer analysis, ping gap
- [`docs/core-flows.md`](./docs/core-flows.md) — Command-level sequence diagrams (new, exec, upload, keep-alive, relay, stop)
- [`docs/google-colab-cli-source-analysis.md`](./docs/google-colab-cli-source-analysis.md) — Full source code architecture reference (v0.5.11)
- [`docs/guides/`](./docs/guides/) — How-to guides (Colab CLI, multi-account, quantization, session monitoring)
- [`docs/reference/`](./docs/reference/) — Technical deep-dives (model gotchas, Kaggle analysis, CUDA dark corners, DL training tricks, AutoDL platform)
- [`.claude/skills/colab-cli/references/gotchas.md`](./.claude/skills/colab-cli/references/gotchas.md) — 22 field-tested gotchas
- [`.claude/skills/colab-cli/references/workflows.md`](./.claude/skills/colab-cli/references/workflows.md) — Full workflow patterns
- [`projects/rl/REPORT_ddpg_vs_td3.md`](./projects/rl/REPORT_ddpg_vs_td3.md) — DDPG vs TD3 benchmark report
- [`index.md`](./index.md) — Full project index with descriptions

## License

MIT
