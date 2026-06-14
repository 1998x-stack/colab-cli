# colab-cli — Project Index

GPU training from the terminal. 38 projects across 6 domains, all trained on free-tier Colab/Kaggle GPUs.

## Projects

### Reinforcement Learning (`projects/rl/`) — 10 projects

| Project | Algorithm | Environment | Key Result |
|---------|-----------|-------------|------------|
| [`alphago`](./projects/rl/alphago/) | MCTS + Policy/Value Networks | 9×9 Go | Neural-guided Monte Carlo Tree Search |
| [`td3-gym`](./projects/rl/td3-gym/) | TD3 | Pendulum-v1 | **-71.76** final = best |
| [`ddpg-gym`](./projects/rl/ddpg-gym/) | DDPG | Pendulum-v1 | -40.95 peak, lost after ep 40 |
| [`ddpg-td3-mujoco`](./projects/rl/ddpg-td3-mujoco/) | DDPG vs TD3 | HalfCheetah, Hopper, Walker2d | Head-to-head MuJoCo comparison |
| [`ddqn-noisy-ram`](./projects/rl/ddqn-noisy-ram/) | Double DQN + NoisyNet | Atari RAM (3 envs) | Prioritized ER for sparse rewards |
| [`ppo-atari-ram`](./projects/rl/ppo-atari-ram/) | PPO + GAE | Atari RAM (63 envs) | Two-tier config system |
| [`ppo-mujoco`](./projects/rl/ppo-mujoco/) | PPO + GAE | MuJoCo continuous | Gaussian policy, vectorized envs |
| [`rl-dqn-atari`](./projects/rl/rl-dqn-atari/) | DQN + Dueling | ALE/Pong-v5 | CNN over stacked grayscale frames |
| [`rl-sac`](./projects/rl/rl-sac/) | SAC + auto entropy | MountainCarContinuous | Automatic temperature tuning |
| [`rl-sarsa-gym`](./projects/rl/rl-sarsa-gym/) | Tabular SARSA | CartPole-v1 | 12-bin discretization (20K states) |

**[DDPG vs TD3 Benchmark Report](./projects/rl/REPORT_ddpg_vs_td3.md)** — 20-point eval table, stability analysis, catastrophic forgetting analysis.

### Computer Vision (`projects/cv/`) — 5 projects

| Project | Model | Dataset | Notes |
|---------|-------|---------|-------|
| [`alexnet_imagenette`](./projects/cv/alexnet_imagenette/) | AlexNet (faithful repro) | Imagenette-160 | 4-experiment ablation study |
| [`vit-cifar10`](./projects/cv/vit-cifar10/) | Vision Transformer (scratch) | CIFAR-10 | 3 configs (baseline/deep/small-patch) |
| [`cnn-cifar10`](./projects/cv/cnn-cifar10/) | 3-block CNN | CIFAR-10 | Conv-BN-ReLU-MaxPool |
| [`cnn-quantization`](./projects/cv/cnn-quantization/) | ResNet-18 | CIFAR-10 | FP32 vs FP16 vs INT8 vs INT4 |
| [`cnn-explainer`](./projects/cv/cnn-explainer/) | CNN + XAI | CIFAR-10 | Grad-CAM, Saliency, Integrated Gradients |

### NLP & LLMs (`projects/nlp/`) — 15 projects

| Project | Task | Model / Approach |
|---------|------|-----------------|
| [`transformer_iwslt`](./projects/nlp/transformer_iwslt/) | De→En Translation | Transformer base, BPE, beam search |
| [`transformer-ln-comparison`](./projects/nlp/transformer-ln-comparison/) | Pre-LN vs Post-LN | Transformer, IWSLT2017, side-by-side comparison |
| [`transformer-kv-cache`](./projects/nlp/transformer-kv-cache/) | KV Cache mechanisms | 4 variants (MHA/MQA/GQA/MLA) + Flash Attention |
| [`nanogpt`](./projects/nlp/nanogpt/) | Char-level LM | nanoGPT on Tiny Shakespeare |
| [`nanochat-colab`](./projects/nlp/nanochat-colab/) | Full-stack ChatGPT clone | 73M GPT (depth=6), tokenizer→pretraining |
| [`s1-t4`](./projects/nlp/s1-t4/) | Test-time scaling | QLoRA Qwen2.5-7B on s1K subset |
| [`seq2seq-t4`](./projects/nlp/seq2seq-t4/) | Seq2Seq with attention | LSTM encoder-decoder |
| [`rnn-imdb`](./projects/nlp/rnn-imdb/) | Sentiment analysis | BiLSTM on IMDB |
| [`word2vec-c4`](./projects/nlp/word2vec-c4/) | Word embeddings | Skip-gram + negative sampling on C4 |
| [`fasttext-pytorch`](./projects/nlp/fasttext-pytorch/) | Efficient text classification | FastText with n-gram features |
| [`rag-fasttext`](./projects/nlp/rag-fasttext/) | Hybrid RAG | BM25 + FastText + FAISS fusion |
| [`hotpotqa-reasoning`](./projects/nlp/hotpotqa-reasoning/) | QA reasoning | CoT vs ReAct, Qwen2.5-7B-AWQ |
| [`text2sql_finetune`](./projects/nlp/text2sql_finetune/) | Text-to-SQL fine-tuning | LoRA Qwen2.5-Coder-7B on medical SQL |
| [`vllm-compare`](./projects/nlp/vllm-compare/) | vLLM benchmarks | 3 models, latency/VRAM comparison |
| [`vllm-rag`](./projects/nlp/vllm-rag/) | RAG pipeline | Qwen2.5-7B-AWQ + ChromaDB + SQuAD v2 |

### Graph Neural Networks (`projects/gnn/`) — 1 project

| Project | Model | Datasets |
|---------|-------|----------|
| [`gnn-citation`](./projects/gnn/gnn-citation/) | 2-layer GCN | Cora, CiteSeer, PubMed |

### Systems & Infrastructure (`projects/systems/`) — 5 projects

| Project | Description |
|---------|-------------|
| [`autoresearch-t4`](./projects/systems/autoresearch-t4/) | Autonomous LLM pretraining research (5-min budget, T4) |
| [`cuda-tutorial`](./projects/systems/cuda-tutorial/) | 7 progressive CUDA kernel tutorials (numba.cuda) |
| [`swe-agent-colab`](./projects/systems/swe-agent-colab/) | SWE-agent on Colab: vLLM + Qwen2.5-7B → fix GitHub issues |
| [`pytorch-transfer-benchmark`](./projects/systems/pytorch-transfer-benchmark/) | CPU-GPU transfer throughput: pinned memory 9.2× speedup |
| [`torch-compile-pipeline`](./projects/systems/torch-compile-pipeline/) | torch.compile pipeline benchmarks and analysis |

### Tutorials (`projects/tutorials/`) — 2 projects

| Project | Description |
|---------|-------------|
| [`ml-tutorial`](./projects/tutorials/ml-tutorial/) | NLP, CV, Audio — fine-tune pretrained transformers |
| [`sklearn-papermill`](./projects/tutorials/sklearn-papermill/) | Parameterized sklearn notebook with papermill on Colab |

---

## Docs

### Guides (`docs/guides/`) — how-to documentation

| Doc | Topic |
|-----|-------|
| [`DeepSeek-Colab CLI 深度使用指南`](./docs/guides/DeepSeek-Colab%20CLI%20深度使用指南.md) | Comprehensive Colab CLI guide (Chinese) |
| [`multi-account-colab`](./docs/guides/multi-account-colab.md) | Multi-account Colab setup |
| [`quantization-guide`](./docs/guides/quantization-guide.md) | Model quantization guide |
| [`session-health-monitoring`](./docs/guides/session-health-monitoring.md) | Colab session health monitoring |
| [`auto-compaction-guide`](./docs/guides/auto-compaction-guide.md) | Context auto-compaction guide |
| [`colab-drivemount`](./docs/guides/colab-drivemount.md) | Google Drive mount on Colab |
| [`colab-keepalive-implementation`](./docs/guides/colab-keepalive-implementation.md) | Colab GPU keepalive implementation guide |
| [`notebook-execution`](./docs/guides/notebook-execution.md) | Notebook execution on Colab |
| [`papermill-colab`](./docs/guides/papermill-colab.md) | Papermill + Colab integration |

### Reference (`docs/reference/`) — technical deep-dives

| Doc | Topic |
|-----|-------|
| [`model-gotchas`](./docs/reference/model-gotchas.md) | Cross-project model + training gotchas |
| [`colab-gpu-keepalive`](./docs/reference/colab-gpu-keepalive.md) | GPU session keepalive root-cause analysis |
| [`core-flows`](./docs/reference/core-flows.md) | Core Colab CLI workflows and patterns |
| [`google-colab-cli-source-analysis`](./docs/reference/google-colab-cli-source-analysis.md) | Google Colab CLI source code analysis |
| [`websocket-stability-analysis`](./docs/reference/websocket-stability-analysis.md) | WebSocket stability analysis |
| [`websocket-stability-china`](./docs/reference/websocket-stability-china.md) | WebSocket stability from China |
| [`kaggle-notebooks-analysis`](./docs/reference/kaggle-notebooks-analysis.md) | Kaggle notebook analysis |
| [`drive-mcp-colab-integration`](./docs/reference/drive-mcp-colab-integration.md) | Drive + MCP + Colab integration |
| [`statusline-config`](./docs/reference/statusline-config.md) | Statusline configuration |
| [`pytorch-cpu-gpu-transfer`](./docs/reference/pytorch-cpu-gpu-transfer.md) | GPU→CPU 9× slowdown root cause (DMA, pinned memory) |
| [`pytorch-cuda-dark-corners`](./docs/reference/pytorch-cuda-dark-corners.md) | Catalog of non-obvious performance pitfalls for T4 benchmarking |
| [`text2sql-finetune`](./docs/reference/text2sql-finetune.md) | Text-to-SQL fine-tuning article (Zhihu) |

### Other docs

| Directory | Contents |
|-----------|----------|
| [`docs/plots/`](./docs/plots/) | Generated plots and figures |
| [`docs/superpowers/`](./docs/superpowers/) | Superpowers plans and specs |
| [`docs/google-workspace-mcp/`](./docs/google-workspace-mcp/) | Google Workspace MCP integration |

---

## Papers (`papers/`)

| Directory | Paper |
|-----------|-------|
| [`s1/`](./papers/s1/) | s1: Simple test-time scaling |
| [`seq2seq/`](./papers/seq2seq/) | Sequence to Sequence Learning with Neural Networks |

---

## Skills (`.claude/skills/`)

| Skill | Purpose |
|-------|---------|
| `colab-cli/` | Colab GPU VM management from terminal |
| `kaggle-cli/` | Kaggle Notebooks GPU training |

---

## Quick Links

- [Landing Page](https://1998x-stack.github.io/colab-cli/)
- [CLAUDE.md](./CLAUDE.md) — Full project conventions and gotchas
- [README.md](./README.md) — Quickstart, proxy setup, account inventory
