# colab-cli Workspace

Machine learning training workspace using [Google Colab CLI](https://pypi.org/project/google-colab-cli/) — provision GPU/TPU VMs from the terminal, run training scripts remotely, and sync artifacts back.

## Setup

```bash
# Install colab CLI
uv tool install google-colab-cli

# Authenticate (OAuth2 is default)
colab --auth oauth2 new   # opens browser for OAuth flow
# or with ADC:
colab --auth adc new
```

## Quick Start

```bash
colab new --gpu T4 -s training                    # provision T4 GPU VM
colab upload train.py train.py                    # upload script
colab exec -s training -f train.py --timeout 300  # run training
colab download -s training /content/checkpoints/ . # pull results
colab stop -s training                            # release VM
```

## Projects

| Project | Description |
|---------|-------------|
| `projects/rl-sac/` | SAC (Soft Actor-Critic) on MountainCarContinuous-v0 |
| `projects/cnn-cifar10/` | CNN classifier on CIFAR-10 |
| `projects/nanogpt/` | NanoGPT training experiments |
| `projects/rl-dqn-atari/` | DQN on Atari environments |
| `projects/vllm-compare/` | vLLM inference benchmarks |
| `projects/vllm-rag/` | RAG pipeline with vLLM |
| `projects/alexnet-imagenette/` | AlexNet on Imagenette |
| `projects/autoresearch-t4/` | Automated ML research on T4 |
| `projects/cuda-tutorial/` | CUDA kernel tutorials |
| `projects/ml-tutorial/` | ML fundamentals |
| `projects/nanochat-colab/` | NanoChat on Colab |
| `projects/rnn-imdb/` | RNN sentiment analysis on IMDB |

## Docs

- [`docs/DeepSeek-Colab CLI 深度使用指南.md`](docs/DeepSeek-Colab%20CLI%20深度使用指南.md) — Comprehensive Chinese-language guide to colab CLI internals (source-code level)
- [`docs/colab-drivemount.md`](docs/colab-drivemount.md) — Google Drive mount guide
- [`docs/multi-account-colab.md`](docs/multi-account-colab.md) — Multi-account management
- [`docs/model-gotchas.md`](docs/model-gotchas.md) — Model training pitfalls
- [`docs/superpowers/`](docs/superpowers/) — Agent skill definitions

## Session Hygiene

- **Always `colab stop` when done** — idle VMs burn compute units
- Free tier VMs auto-terminate after ~2-4 hours
- `colab sessions` lists active sessions; `[?]` = orphaned server-side
- `colab log -s <name> -o report.ipynb` exports session history as notebook

## Key Gotchas

- `colab exec -f` takes **relative paths** only (uploads from local, runs remotely)
- Detached subprocess training needs `PYTHONUNBUFFERED=1` + `start_new_session=True`
- Unrecognized `--gpu` values silently fallback to A100 then fail with 400
- `colab run` is ideal for one-shot jobs; it provisions → runs → tears down automatically
