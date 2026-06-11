# SWE-Agent on Colab: Faithful Mini-Port with vLLM + Qwen2.5-7B-Instruct-AWQ

## Overview

Port SWE-agent's core agent loop and ACI (Agent-Computer Interface) to run on Colab T4 with vLLM serving Qwen2.5-7B-Instruct-AWQ. Evaluate on 2-3 filtered SWE-bench Lite tasks.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Task Config                       │
│  repo_url, commit, problem_statement, test_cmd    │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│                 SWE-Agent Loop                     │
│                                                    │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐ │
│  │ System + │   │  Model   │   │    Parse       │ │
│  │ Instance │──▶│ (vLLM)   │──▶│ Thought+Action │ │
│  │ Template │   │          │   │                │ │
│  └──────────┘   └──────────┘   └───────┬───────┘ │
│                                        │          │
│  ┌──────────────────────────────────────▼───────┐ │
│  │              Handle Action                    │ │
│  │  ┌──────┐  ┌──────────────────┐  ┌────────┐ │ │
│  │  │Block?│  │Execute in Env    │  │Submit? │ │ │
│  │  │retry │  │(bash/edit tools) │  │extract │ │ │
│  │  └──────┘  └────────┬─────────┘  │patch   │ │ │
│  └──────────────────────┼────────────└────────┘ │ │
│                         │                        │ │
│  ┌──────────────────────▼───────────────────────┐ │
│  │  Observation + State → append to history      │ │
│  │  (truncate if >N chars, format with template) │ │
│  └──────────────────────┬───────────────────────┘ │
│                         │                          │
│                         └──────▶ loop back ───────┘│
└──────────────────────────────────────────────────┘
```

## File Layout

```
projects/swe-agent-colab/
├── agent.py          # Agent loop, forward, handle_action, submit
├── tools.py           # ACI commands: bash, str_replace_editor, submit
├── config.yaml        # System/instance templates, tool settings
├── models.py          # vLLM client (OpenAI-compatible API)
├── tasks.py           # Task definitions (repo, commit, problem, test)
├── environment.py     # Git clone, checkout, apply patch, run tests
├── launch.py          # Colab bootstrap: pip install, start vLLM, spawn agent
├── run.py             # Local entry: load tasks, run agent, save metrics
├── evaluate.py        # Run tests, compute pass@1, generate charts
└── check_progress.py  # Monitor vLLM + agent health
```

Data flow: `run.py` → loads `tasks.py` → for each task: init `environment.py` (clone repo) → init `agent.py` with `tools.py` + `models.py` (vLLM) → run agent loop → save trajectory → `evaluate.py` → `metrics.json` + PNGs.

## ACI Tools

Three tools matching the paper's design:

### 1. `bash <command>`
Execute any bash command. Returns stdout+stderr. Timeout 30s. Blocked: interactive programs (vim, nano, less, python/python3 standalone, bash/sh standalone). Exit code shown in observation.

### 2. `str_replace_editor <command> <path> [args...]`

| Sub-command | Args | Behavior |
|---|---|---|
| `view` | `path`, `[view_range]` | File: `cat -n` output. Dir: `ls -R` 2 levels deep |
| `create` | `path`, `file_text` | Create new file, fail if exists |
| `str_replace` | `path`, `old_str`, `new_str` | Replace exact match of `old_str` with `new_str`. Fail if not unique |
| `insert` | `path`, `insert_line`, `new_str` | Insert `new_str` after `insert_line` |
| `undo_edit` | `path` | Revert last edit to file at `path` |

### 3. `submit`
Generates `git diff` patch → writes `/root/model.patch` → exits with submission.

### Tool config
- Parse via function-calling (vLLM OpenAI-compatible endpoint). Regex fallback for thought/action parsing.
- Blocklist: `vim, vi, emacs, nano, less, tail -f` (prefix), `python, python3, bash, sh` (standalone)
- Observation truncation: 50k chars max with `<response clipped><NOTE>...`

## Agent Loop & Prompts

### System Template
```
You are a helpful assistant that can interact with a computer to solve tasks.
You have access to the following functions:
{command_docs}
```

### Instance Template
```
I've uploaded a python code repository in /testbed. Consider the following issue:

<issue>
{problem_statement}
</issue>

Follow these steps:
1. Explore the repository to understand the relevant code
2. Create a script to reproduce the error and run it with `bash python <script.py>`
3. Edit the source code using str_replace_editor to fix the issue
4. Re-run the reproduction script to confirm the fix
5. Think about edge cases and ensure the fix handles them
6. Submit your changes with the submit command
```

### Observation Template
```
OBSERVATION:
{observation}
```
Truncated variant: `{observation[:50000]}<response clipped><NOTE>Observations should not exceed 50000 characters...</NOTE>`

### Agent Loop (pseudocode)
```python
def run(env, problem):
    history = [system_msg, instance_msg]
    for step in range(max_steps=30):
        output = vllm.chat(messages=history, tools=aci_tools)
        thought, action = parse(output)

        # Error handling (up to 3 requeries each)
        if blocked_action(action) or syntax_error(action) or malformed(output):
            history += [assistant_msg, error_msg]
            continue

        observation = env.execute(action)
        history += [assistant_msg(thought, action), observation_msg(observation)]

        if is_submission(observation):
            return env.read_file("/root/model.patch")
    return None
```

### Error handling (max 3 requeries each)
- **Blocked action:** "Operation 'X' is not supported by this environment."
- **Bash syntax error:** Show `bash -n` output.
- **Malformed output:** "Could not parse action. Use function calling format."

### Key parameters
- `max_steps`: 30
- `execution_timeout`: 30s per command
- `max_consecutive_timeouts`: 3 → autosubmit
- `max_observation_length`: 50,000 chars
- Cache control: last 2 messages

## Experiments

### Tasks
2-3 SWE-bench Lite tasks, filtered for Colab compatibility:
- Pure Python repos (no C extensions, no Docker)
- Light test deps (pytest only)
- Single-file or few-file fixes
- Clear problem statements

### Experiment configs

| Config | Tools | Purpose |
|--------|-------|---------|
| Full ACI | bash + str_replace_editor + submit | Main experiment |
| Bash-only (optional) | bash + submit | Baseline |

### Per-task metrics
```json
{
  "task_id": "django__django-11066",
  "resolved": true,
  "steps_taken": 8,
  "total_tokens": 12450,
  "prompt_tokens": 8900,
  "completion_tokens": 3550,
  "total_time_seconds": 234.5,
  "model_queries": 9,
  "errors": {"blocked": 0, "syntax": 0, "malformed": 1},
  "patch": "..."
}
```

### Output files

| File | Content |
|------|---------|
| `metrics.json` | All per-task metrics + summary |
| `trajectory_<task>.json` | Full step-by-step history |
| `results.png` | Bar chart: resolved/steps/tokens per task |
| `token_allocation.png` | Prompt vs completion tokens per step |
| `timeline.png` | Step duration per task |
| `agent.log` | Full debug log |

### Success criteria
- Agent loop runs end-to-end without crashes
- At least 1/3 tasks resolved (Qwen2.5-7B is small; paper's 12.5% was GPT-4)
- All output files generated

## Implementation notes

- vLLM serves via OpenAI-compatible API on port 8000
- Qwen2.5-7B-Instruct-AWQ fits in ~5GB VRAM on T4 (16GB total)
- Agent and task environment run on the same VM (no Docker needed)
- Tasks cloned to `/testbed/` inside the VM
- Detached execution via `launch.py` (pip install + start vLLM + spawn agent as subprocess)
- Check progress via `check_progress.py` (vLLM health, agent process status, log tail)
