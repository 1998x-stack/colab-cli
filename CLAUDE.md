# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A workspace for running ML training workloads on Google Colab via the `colab` CLI. The primary project is SAC (Soft Actor-Critic) reinforcement learning on `MountainCarContinuous-v0`.

The `colab` CLI is installed separately as a system tool (`uv tool install google-colab-cli`). The skill at `.claude/skills/colab-cli/SKILL.md` provides detailed usage reference.

## Workflow pattern

```bash
colab new --gpu T4 -s training          # provision VM
colab upload launch.py launch.py        # upload launcher script
colab exec -f launch.py --timeout 120   # start training (detached)
colab exec -f check_progress.py         # check progress
colab download /content/checkpoints/ .  # pull artifacts
colab stop -s training                  # release VM
```

## Detached training (critical gotchas)

- **stdout is buffered in subprocess**: Always set `PYTHONUNBUFFERED=1` in env and use `python -u` when spawning background processes. Without this, log files appear empty even though training runs.
- **`start_new_session=True`**: Required for `subprocess.Popen` to survive kernel exec timeouts. Without it the child gets SIGHUP when `colab exec` disconnects.
- **`colab exec -f` takes relative paths only**: Upload to `/content/foo.py` but run with `-f foo.py`. Absolute paths fail with FileNotFoundError.
- **VMs auto-terminate after ~2-4 hours** on free tier. Checkpoints and downloaded files are the only persistence. `colab run` (without `--keep`) self-cleans regardless.

## Project: SAC MountainCar (`projects/rl-sac/`)

| File | Purpose |
|------|---------|
| `sac_mountaincar.py` | SAC agent + training loop. 500 episodes, checkpoints to `/content/checkpoints/` every 50 episodes, tensor logging every 10. |
| `launch.py` | Runs on Colab VM. Installs `gymnasium`, spawns `sac_mountaincar.py` as detached subprocess with unbuffered output to `/content/sac_train.log`. |
| `check_progress.py` | Runs on Colab VM. Checks if training process is alive (pgrep), tails last 15 log lines, lists checkpoint files with sizes. |

### Training config (defaults in `sac_mountaincar.py`)

LR 3e-4, hidden dim 256, replay buffer 1M, batch 256, start steps 10k, gamma 0.99, tau 0.005, automatic entropy tuning. Saves best model + periodic checkpoints to `/content/checkpoints/`. Resumes from latest checkpoint if one exists.

## Session hygiene

- `colab sessions` lists server-side assignments and prunes stale local entries. Orphans show as `[?]`.
- `colab status` shows hardware, IDLE/BUSY, and last execution output.
- Idle VMs burn compute units — always `colab stop` when done.
- Accelerator availability is tier-gated. `--gpu T4` is most reliable on free tier. TPUs are usually rejected without Pro. Unrecognized `--gpu` values silently fall back to A100 and then fail.
