# SWE-Agent Colab

Deploys a [SWE-agent](https://swe-agent.com)-style coding agent on Google Colab. The agent uses a local vLLM server (Qwen2.5-7B-Instruct-AWQ) to autonomously fix GitHub issues by exploring code, editing files, and running tests.

## Usage

```bash
# Colab deployment (A100 or high-memory GPU recommended)
cb launch.py --timeout 1800
```

The pipeline:
1. Install dependencies (`openai`, `vllm`, `pyyaml`, `jinja2`, `matplotlib`)
2. Start vLLM with Qwen2.5-7B-Instruct-AWQ (requires ~16GB+ VRAM)
3. Wait for vLLM health check
4. Run the agent on 3 predefined GitHub issues

## Architecture

- **Model:** Qwen2.5-7B-Instruct-AWQ via vLLM (OpenAI-compatible API)
- **Tools:** `bash`, `str_replace_editor`, `submit` (ACI tool interface)
- **Agent loop:** Max 30 steps, 3 requeries per step, 30s execution timeout
- **Environment:** Clone + checkout commit, shell execution, git patch management

## Task instances

| Task ID | Repository | Issue |
|---------|-----------|-------|
| astropy__astropy-14365 | astropy/astropy | Compound model constraint propagation |
| django__django-11066 | django/django | Missing `confirm_login_allowed` call |
| pylint-dev__pylint-4970 | pylint-dev/pylint | False positive `used-before-assignment` |

## Current results

| Metric | Value |
|--------|-------|
| Total tasks | 3 |
| Resolved | 1 (astropy) |
| Pass rate | 33% |

## Gotchas

- Requires a GPU with at least 16 GB VRAM (T4 is marginal; A100 or L4 recommended).
- vLLM server startup takes 5-15 minutes (model download + CUDA graph compilation).
- The agent runs autonomously for up to 30 steps; the full run can take 30-60 minutes.
- Monitor progress with `check_progress.py` or via the heartbeat file.
