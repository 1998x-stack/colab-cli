---
name: colab-cli
description: >
  Use when working with Google Colab from the terminal — provisioning GPU/TPU VMs,
  running code remotely on Colab sessions, uploading/downloading files, executing
  long-running training jobs with nohup, or debugging Colab session issues. Triggers
  on mentions of Colab, colab CLI, Google Colab runtimes, running Python on Colab VMs,
  or needing cloud GPU/TPU compute from the command line. Also trigger when the user
  wants to background a training run, check on remote training progress, or has issues
  with `colab` commands.
---

# Colab CLI

Command-line interface for Google Colab — provision GPU/TPU VMs, run code remotely, and manage files from the terminal.

## Mental model

Colab sessions are **ephemeral Linux VMs** running Jupyter kernels. Free-tier GPU sessions last ~2-4 hours, then vanish with all files. There is no persistent storage — everything must be downloaded before the session ends.

Key distinctions that trip people up:

- **`colab exec -f` reads LOCAL files** from your CWD and sends them to the VM for execution. It does NOT run files already on the VM. Upload is only needed for files that your exec'd script spawns as subprocesses.
- **The VM's working directory is `/content/`.** Uploaded files land there. `colab exec -f` is relative to your local CWD, not `/content/`.
- **The kernel WebSocket is flaky.** For anything that takes >30s (pip install, model download), spawn a detached subprocess via `start_new_session=True` and exit immediately. The exec returns, the work continues.
- **Sessions die silently.** No warning, no recovery. Download checkpoints during the run, not at the end.

## Quick reference

```bash
colab new --gpu T4 -s <name>           # create GPU session (T4, L4, G4, H100, A100)
colab new --tpu v5e1 -s <name>         # TPU (v5e1, v6e1) — may need quota
colab new -s <name>                    # CPU fallback
colab run --gpu T4 <script.py>         # one-shot: provision, run, teardown — no session mgmt
colab sessions                         # list active sessions
colab status [-s <name>]               # show session status
colab ls [-s <name>]                   # list files in session
colab upload <local> /content/<name>   # upload file (always use absolute remote path)
colab download <remote> <local>        # download file (single files only, tar directories first)
colab exec -f <script.py> [--timeout]  # execute LOCAL Python file (reads from CWD, sends to VM)
colab whoami                           # show active account email
colab url [-s <name>]                  # get browser URL for session
colab stop [-s <name>]                 # stop session
```

## Proxy setup (REQUIRED from China)

Google Colab APIs are blocked in mainland China. Route through the local Clash/Meta proxy (mixed-port 7890). **Prefix every `colab` command with:**

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

- `HTTPS_PROXY`/`HTTP_PROXY` covers REST API calls (`colab.research.google.com`, `colab.pa.googleapis.com`)
- `ALL_PROXY=socks5://` covers WebSocket kernel connections (`*.colab.dev`)
- **Without these, every command fails** with `SSLError: UNEXPECTED_EOF_WHILE_READING`

**If `colab exec` gets `RuntimeError: Connection was lost`**, the WebSocket can't handle SOCKS5. Keep the HTTPS_PROXY/HTTP_PROXY but bypass the WebSocket domains:

```bash
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

Which variant works changes per session — flip and retry. `colab sessions`/`colab new`/`colab stop` always use the proxy. Only `colab exec`/`colab download`/`colab upload` might need `no_proxy`.

## Multi-account setup

The `colab` CLI does not natively support multiple accounts — the OAuth2 token path (`~/.config/colab-cli/token.json`) is hardcoded. The workaround is **separate `$HOME` directories** for each account, which fully isolates token, sessions, settings, logs, and history.

### This machine's accounts

Four aliases are configured in `~/.zshrc` (proxy included):

| Alias | Account | HOME |
|-------|---------|------|
| `colab` | hackxie1998@gmail.com | default `~` |
| `cb` | stefaniehu929@gmail.com | `~/colab-accounts/account-b` |
| `cc` | xbetterdetermine@gmail.com | `~/colab-accounts/account-c` |
| `clb` | xieminghack@gmail.com | `~/colab-accounts/account-clb` |

```bash
# Fully interchangeable with the standard colab CLI:
cb new --gpu T4 -s training
cb exec -f train.py --timeout 120
cb sessions
cb stop -s training

cc new --gpu T4 -s inference
cc exec -f infer.py

clb new --gpu T4 -s experiment
clb exec -f run.py --timeout 120
```

**Verification:** `colab whoami` / `cb whoami` / `cc whoami` / `clb whoami` shows which account is active.

**How it works:** All `colab` state paths derive from `$HOME` (`~/.config/colab-cli/`, `~/.colab-cli-oauth-config.json`). Each alias overrides `HOME` to point at an isolated directory tree. The proxy env vars (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`) are baked into each alias so they work from any shell.

**Adding more accounts:** See `docs/multi-account-colab.md` for the full guide.

## Session lifecycle

Colab sessions are ephemeral. Free-tier GPU (T4) sessions last ~12-15 minutes before auto-termination. Files and checkpoints survive within a session but vanish when the session ends. Use `colab download` to pull important artifacts back.

Check session health after any connectivity error — transient SSL/connection errors happen and don't necessarily mean the session is dead:

```bash
colab sessions && colab status
```

## Executing code

### Running a script (`colab exec -f`)

The `-f` flag reads a Python file from your **local filesystem** (relative to CWD) and sends it to the VM for execution. It does NOT run files already on the VM:

```bash
colab exec -f script.py --timeout 120     # sends ./script.py to VM, executes it
```

The file is transmitted to the kernel — no separate upload needed. But if your script spawns subprocesses that reference other files (e.g., `subprocess.Popen(["python", "/content/worker.py"])`), those files must be uploaded separately.

### Inline code via stdin

For quick one-liners without creating files, pipe Python to stdin:

```bash
echo 'print("hello")' | colab exec --timeout 10
echo 'import torch; print(torch.cuda.is_available())' | colab exec --timeout 10
```

**There is no `-c` flag.** `colab exec -c "..."` fails with "No such option: -c". Always use stdin pipe for inline code.

**Avoid f-strings and special characters in stdin pipes** — the shell interprets `$`, `\\`, and `{}` before Python sees them. For anything beyond simple expressions, use a script file with `colab exec -f` instead.

### Background / nohup execution

For long-running training jobs, use `scripts/launch_proxy.py` — a template that pip-installs dependencies and spawns your training script as a detached subprocess (survives after `colab exec` returns, unbuffered output).

**Before running, edit the template** to set your script name and dependencies:

```python
# In scripts/launch_proxy.py:
SCRIPT = "train.py"              # Your script (already on VM at /content/)
DEPS = ["torch", "transformers"] # pip packages to install
LOG = "/content/train.log"       # Where stdout/stderr goes
```

**Workflow:**

```bash
# 1. Provision
colab new --gpu T4 -s training

# 2. Upload your training script
colab upload train.py /content/train.py

# 3. Launch (pip install + spawn, auto-detects proxy if running)
colab exec -f scripts/launch_proxy.py --timeout 120

# 4. Check progress (proxy health, process status, log tail, checkpoints)
colab exec -f scripts/check_progress.py --timeout 15
```

**`scripts/check_progress.py`** checks proxy health + process alive + log tail + checkpoints. Override defaults via env vars: `CHECK_SCRIPT`, `CHECK_LOG`, `CHECK_CKPT`.

**About VM-side proxy:** Colab VMs have excellent direct internet from GCP (Google, HuggingFace, PyPI, GitHub all HTTP 200). `proxy.yaml` SS servers are unreachable from GCP (`bit-*.kunlun03dns.com` → `103.181.164.x` times out). `scripts/vm-proxy-bootstrap.py` tests reachability before starting; exits cleanly if unreachable. Don't bother with VM proxy unless you hit a specific geo-block.

### Launching without the template

If you prefer a minimal launcher, the essential pattern is:

```python
import subprocess, sys, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium", "-q"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open("/content/train.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
print(f"OK. PID={proc.pid} log=/content/train.log")
```

The critical bits: `PYTHONUNBUFFERED=1` + `python -u` (unbuffered output), `start_new_session=True` (survives exec timeout), and pip install happens before spawning.

## One-shot execution (`colab run`)

For scripts that don't need monitoring, `colab run` provisions a fresh VM, runs the script, and auto-teardowns:

```bash
colab run --gpu T4 script.py
```

No session management, no uploads, no teardown. Best for batch jobs, benchmarks, or CI-style workflows. The VM is destroyed after the script completes.

## Gotchas

These are field-tested patterns that differ from what you'd expect. Read `references/gotchas.md` for the full list with detailed explanations. The critical ones:

1. **Proxy required from China.** Set `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` before every command. See Proxy setup section above.
2. **`colab exec -f` reads LOCAL files (relative to CWD), not remote VM files.** Upload is only needed for scripts spawned as subprocesses by the exec'd script. `cd` to the right directory before `colab exec -f`.
3. **Use detached bootstrap for any workflow with pip install or sustained operations.** `colab exec` WebSocket drops during runs >30s. Spawn a bootstrap via `start_new_session=True` that handles everything — the exec returns immediately. See `references/gotchas.md` for the pattern.
4. **CUDA version mismatch on Colab T4.** VM has CUDA 12.8 with PyTorch 2.11.0+cu128. Latest vLLM's default wheel requires CUDA 13. Install vLLM with `--extra-index-url https://download.pytorch.org/whl/cu128` (not `--index-url`). See GPU/CUDA section in `references/gotchas.md`.
5. **stdout is buffered in subprocess.** Set `PYTHONUNBUFFERED=1` and use `python -u` when spawning background jobs, or logs stay empty.
6. **Only 1 GPU session per account on free tier.** Provisioning a second GPU on the same account raises `TooManyAssignmentsError`. Use the multi-account aliases (`cb`, `cc`, `clb`) to run parallel GPU sessions across accounts.
7. **Sessions get pruned.** After ~2-4h of idle or total runtime, the session disappears. Can happen in <5 min after connection errors. Download checkpoints regularly.
8. **`colab download` doesn't do directories.** Tar on the VM first: `tar -czf /content/out.tar.gz -C /content dir/`.
9. **SSL errors are often transient.** Re-check `colab sessions` — background processes may still be alive.
10. **Upload: use absolute remote paths.** `colab upload local.py /content/train.py` — relative paths may silently fail.
11. **Colab VMs have good direct internet (GCP).** Google, HuggingFace, PyPI, GitHub all reachable directly. `proxy.yaml` SS servers are unreachable from GCP.
12. **`colab exec` has NO `-c` flag.** Use stdin pipe for inline code: `echo '...' | colab exec`. Avoid f-strings in stdin pipes — use script files instead. See `references/gotchas.md`.
13. **numba.cuda: `cuda.grid(2)` returns (x, y) = (col, row).** Map carefully in 2D kernels. Use `cuda.to_device()` for explicit device arrays. See GPU/CUDA section in `references/gotchas.md`.
14. **Free tier sessions can die in <30 minutes.** Fix bugs locally — provision + upload + launch immediately after. See Session lifetime in `references/gotchas.md`.

## Hardware availability

Request accelerators with `--gpu` or `--tpu`. Availability depends on your Colab subscription tier:

- **Free**: T4 GPU usually available. TPU and higher-end GPUs often rejected.
- **Pro/Pro+**: L4, G4, sometimes H100/A100. TPU v5e1 typically works.

Always have a CPU fallback plan. If the accelerator is rejected, the CLI prints a clear error — try a different one or omit the flag for CPU.

## File paths

The Colab VM's working directory is `/content/`. Uploaded files with relative paths land there. Use `colab ls` to verify what's on the VM.
