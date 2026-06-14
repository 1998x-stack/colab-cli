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

Colab sessions are **ephemeral Linux VMs** running Jupyter kernels. Official free-tier limits: 12h max session.

The keep-alive daemon (auto-spawned by `colab new`, calls `KeepAliveAssignment` RPC every 60s) is **broken due to an IAM deadlock** — it dies 61 seconds after every session creation. See `docs/colab-gpu-keepalive.md` for the full root-cause analysis.

The **WebSocket connection** (`colab exec`) is the primary liveness signal. While the WebSocket stays open, the session survives — even with a dead keep-alive daemon. The session dies ~2-3 minutes after the last WebSocket closes. From China, WebSocket stability is ~8-12 minutes per connection (see `docs/websocket-stability-china.md`).

All `/content/` files vanish when the session ends — download or persist to Drive before that.

**Two independent network paths:**

- **REST** (`colab.pa.googleapis.com`): `colab new`, keep-alive, `colab stop`. Short-lived HTTPS, goes through `requests` proxy auto-detection.
- **REST** (`*.prod.colab.dev`): `colab upload`, `colab download`. HTTPS PUT/GET via Jupyter Contents API. NOT WebSocket — same domain, different transport.
- **WebSocket** (`*.prod.colab.dev`): `colab exec`. Long-lived WSS, `websocket-client` does NOT pass proxy params — the root cause of most disconnects.

Key distinctions that trip people up:

- **`colab exec -f` reads LOCAL files** from your CWD and sends them to the VM for execution. It does NOT run files already on the VM. Upload is only needed for files that your exec'd script spawns as subprocesses.
- **The VM's working directory is `/content/`.** Uploaded files land there. `colab exec -f` is relative to your local CWD, not `/content/`.
- **The kernel WebSocket is flaky through proxy.** For anything that takes >30s (pip install, model download), spawn a detached subprocess via `start_new_session=True` and exit immediately. The exec returns, the work continues.
- **Sessions die silently.** No warning, no recovery. Write checkpoints to Drive during the run, not after.

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
colab drivemount [-s <name>]           # mount Google Drive at /content/drive
colab whoami                           # show active account email
colab url [-s <name>]                  # get browser URL for session
colab stop [-s <name>]                 # stop session
```

## Proxy setup (REQUIRED from China)

Google Colab APIs are blocked in mainland China. The two-path architecture (see Mental model above) means REST and WebSocket need different proxy treatment.

**Try this first** — REST through SOCKS5, WebSocket direct:

```bash
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

**If WebSocket direct fails**, flip — both paths through proxy (WebSocket treated as HTTP CONNECT tunnel):

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

Which variant works changes per session — flip and retry. `colab sessions`/`colab new`/`colab stop` are REST-only to `colab.pa.googleapis.com` (always use proxy). `colab exec`/`colab upload`/`colab download` all go to `*.prod.colab.dev` — they are affected by `no_proxy` because of the shared domain, even though upload/download use REST while exec uses WebSocket.

See `docs/websocket-stability-china.md` for the full root-cause analysis.

## Multi-account setup

The `colab` CLI does not natively support multiple accounts — the OAuth2 token path (`~/.config/colab-cli/token.json`) is hardcoded. The workaround is **separate `$HOME` directories** for each account, which fully isolates token, sessions, settings, logs, and history.

### This machine's accounts

Six aliases are configured in `~/.zshrc` (proxy included):

| Alias | Account | HOME |
|-------|---------|------|
| `colab` | hackxie1998@gmail.com | default `~` |
| `cb` | stefaniehu929@gmail.com | `~/colab-accounts/account-b` |
| `cc` | xbetterdetermine@gmail.com | `~/colab-accounts/account-c` |
| `clb` | xieminghack@gmail.com | `~/colab-accounts/account-clb` |
| `clab` | xieminghacker@gmail.com | `~/colab-accounts/account-clab` |

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

clab new --gpu T4 -s job
clab exec -f run.py --timeout 120
```

**Verification:** `colab whoami` / `cb whoami` / `cc whoami` / `clb whoami` / `clab whoami` shows which account is active.

**How it works:** All `colab` state paths derive from `$HOME` (`~/.config/colab-cli/`, `~/.colab-cli-oauth-config.json`). Each alias overrides `HOME` to point at an isolated directory tree. The proxy env vars (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`) are baked into each alias so they work from any shell.

**Adding more accounts:** See `docs/multi-account-colab.md` for the full guide.

### Account status files

Two JSON files in the repo root track multi-account state at a glance — check them before provisioning to pick a healthy account.

**`.colab-gpu-status.json`** — GPU quota availability per account:

```json
{
  "checked_at": "ISO timestamp",
  "accelerator": "T4",
  "proxy_config": "B",
  "accounts": {
    "<alias>": {
      "email": "...",
      "gpu_available": true | false,
      "error": null | "TooManyAssignmentsError (412) ..."
    }
  },
  "summary": { "total": 6, "available": 5, "exhausted": 1 }
}
```

**`.colab-session-status.json`** — active session counts per account:

```json
{
  "checked_at": "ISO timestamp",
  "accounts": {
    "<alias>": {
      "email": "...",
      "gpu_sessions": 0,
      "cpu_sessions": 0,
      "total": 0,
      "stale_pruned": 0
    }
  },
  "summary": { "total_accounts": 6, "total_gpu_sessions": 0, "total_cpu_sessions": 0, "total_sessions": 0 }
}
```

**Regenerate:** Run `colab sessions` on each account to refresh session status. GPU status needs a `colab new --gpu T4` probe per account (stop the test session immediately after). Both files are always read from disk — caching is OK across short intervals but recheck when a previous attempt failed with 412.

## Session lifecycle

Colab sessions are ephemeral. Official free-tier limits: 12h max session, ~90min idle timeout. GPU quota is dynamic — heavy use triggers 12-24h cooldown before GPU becomes available again.

**The 12-15 min effective window** observed from China is WebSocket disconnection through the proxy, NOT Colab killing the session. The keep-alive daemon (auto-spawned by `colab new`, calls `KeepAliveAssignment` RPC via REST every 60s, max 24h) prevents idle timeout. But it does nothing for the exec WebSocket — those are separate network paths.

**Failure modes:**
- **Keep-alive daemon always dead** — IAM deadlock (403 `USER_PROJECT_DENIED`), daemon exits at T+61s every session. Session relies on WebSocket for liveness.
- WebSocket handshake failure (~20-30% of exec attempts) — proxy can't establish WSS tunnel
- WebSocket mid-exec disconnect — NAT timeout or GFW RST, exec hangs then `TimeoutError`
- Session death ~2-3 min after last WebSocket closes (no keep-alive backup)
- GPU quota exhausted — `TooManyAssignmentsError`, switch accounts or wait 12-24h
- Session pruned after connection errors — `[colab] Pruned 1 stale local session(s)`

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

# 3. Launch (pip install + spawn detached, uses direct connection — Colab VMs have good GCP internet)
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

## Gotchas (top criticals)

Read `references/gotchas.md` for the full list (45 field-tested items). The ones that will waste the most time if missed:

1. **Proxy required from China.** REST and WebSocket use different network paths with different proxy behavior. Use the two-config flip pattern from Proxy setup above.
2. **`colab exec -f` reads LOCAL files (relative to CWD), not remote VM files.** Upload is only needed for scripts spawned as subprocesses by the exec'd script.
3. **Use detached bootstrap for any workflow with pip install or sustained operations.** `colab exec` WebSocket drops during runs >30s. Spawn via `start_new_session=True` — the exec returns immediately.
4. **Empty logs do NOT mean the job is stuck.** stdout to file via subprocess can buffer despite `PYTHONUNBUFFERED=1`. Verify with `nvidia-smi` or check for side effects (files appearing, GPU utilization) before assuming a job is dead.
5. **stdout is buffered in subprocess.** Set `PYTHONUNBUFFERED=1` and use `python -u` when spawning background jobs, plus `flush=True` on all print() calls.
6. **Only 1 GPU session per account on free tier.** Use the multi-account aliases (`cb`, `cc`, `clb`) for parallel GPU sessions.
7. **First Colab session rarely produces useful training.** Data download + CUDA JIT = 7-10 min overhead. First session dies before completing an epoch. Second session (data cached) works normally.
8. **`colab download` doesn't do directories.** Tar on VM first: `tar -czf /content/out.tar.gz -C /content dir/`.
9. **REST API survives WebSocket drops.** When `colab exec` returns 404/401, session is usually still alive — use `colab download` as fallback monitoring.
10. **`colab upload` creates a FILE not a directory** when the path doesn't exist. Upload flat to `/content/` root, create dirs via exec.

## WebSocket stability

The root cause of most `colab exec` failures from China: `KernelWebSocketClient._run_websocket()` calls `run_forever()` without proxy parameters, `ping_interval=60` races with NAT timeouts, and `reconnect_interval=0` means no auto-reconnect. See `docs/websocket-stability-china.md` for the full root-cause analysis.

**The fix:** Detached bootstrap — exec returns in seconds, training survives all WebSocket drops (see Executing code > Background / nohup execution above).

## Checkpoint persistence

VM-local files (`/content/*`) vanish when the session ends. Two strategies:

**P0: Drive mount (recommended).** `colab drivemount` → train.py writes checkpoints to `/content/drive/MyDrive/colab-checkpoints/<project>/`. VM→Drive goes over Google internal network, bypassing China proxy entirely.

```bash
colab new --gpu T4 -s training
colab drivemount -s training
# train.py checkpoint path: /content/drive/MyDrive/colab-checkpoints/my-project/ckpt_epoch5.pt
```

**P1: Manual tar+download.** For projects that can't use Drive mount — cron-triggered download of checkpoint tars via `colab download`.

See `docs/drive-mcp-colab-integration.md` for MCP server integration and multi-account Drive management.

## Kaggle Notebooks (complementary GPU)

Kaggle's push model (`kaggle kernels push` is a single REST call) avoids Colab's WebSocket problem entirely. 30h/week GPU (P100 or T4 x2), transparent quota counter.

Use when Colab is unreliable or you need longer training sessions. Key commands:

```bash
kaggle kernels push -p ./project-dir   # push + run (REST, no long connection)
kaggle kernels status user/slug        # check status
kaggle kernels output user/slug -p ./  # download results
```

Script mode (`kernel_type: "script"` in `kernel-metadata.json`) runs plain `.py` files — no notebook conversion needed. See `docs/kaggle-notebooks-analysis.md` for full comparison, integration strategy, and MCP server options.

## Hardware availability

Request accelerators with `--gpu` or `--tpu`. Availability depends on your Colab subscription tier:

- **Free**: T4 GPU usually available. TPU and higher-end GPUs often rejected.
- **Pro/Pro+**: L4, G4, sometimes H100/A100. TPU v5e1 typically works.

Always have a CPU fallback plan. If the accelerator is rejected, the CLI prints a clear error — try a different one or omit the flag for CPU.

## Training outputs — logs, plots, metrics

Every training script should produce three structured artifacts for glance-and-decide monitoring. Use `scripts/log_utils.py` and `scripts/plot_utils.py` for reusable implementations.

### Output directory structure

```
<out_dir>/
├── logs/train.log          # Timestamped training log
├── metrics.csv             # Per-epoch structured metrics
├── pngs/training_curves.png  # Multi-panel visualization
├── checkpoints/            # Model checkpoints (excluded from cron fetch)
└── summary.json            # Final run metadata
```

### log_utils.py — reusable logging

```python
from log_utils import Logger, MetricsCSV, SummaryJSON, detect_output_dir, setup_output_dirs

# Auto-detect environment
out_dir = detect_output_dir("my-project")
setup_output_dirs(out_dir)

# Timestamped log to file + stdout
logger = Logger(f"{out_dir}/logs/train.log")
logger.log("Training started")
logger.log(f"Ep 1/5 | Batch 100 | loss=1.23 | avg100=1.35")

# Structured CSV — header written on creation, rows appended
csv = MetricsCSV(f"{out_dir}/metrics.csv",
                 ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                  "elapsed_s", "lr"])
csv.write_row(epoch=1, train_loss=1.23, train_acc=0.45,
              test_loss=1.34, test_acc=0.50, elapsed_s=180, lr=0.089)

# Final summary
summary = SummaryJSON(f"{out_dir}/summary.json")
summary.write({"test_acc": 0.87, "epochs_completed": 5, "total_time_s": 900})
```

### plot_utils.py — reusable visualization

```python
from plot_utils import plot_loss_acc, plot_rl_progress, plot_loss

# Classification/regression (4-panel: loss, accuracy, LR, distribution)
plot_loss_acc(metrics, f"{out_dir}/pngs/training_curves.png",
              title="My Model — Dataset", size_label="small")

# RL training (4-panel: reward, episode length, exploration, Q-values)
plot_rl_progress(metrics, f"{out_dir}/pngs/rl_progress.png",
                 title="TD3 — HalfCheetah", solved_threshold=10000)

# Minimal single-panel loss plot
plot_loss(metrics, f"{out_dir}/pngs/loss.png", title="Loss")
```

### Log format convention

Per-N-batches, one self-contained line:
```
[HH:MM:SS] Ep 1/5 | Batch 1000 | loss=1.3625 | avg100=1.3677 | lr=0.089451 | elapsed=97s
```

Evaluation lines:
```
[HH:MM:SS]   -- Eval @ batch 1000: test_loss=1.3660 | test_acc=0.4904
```

Epoch-end lines:
```
[HH:MM:SS] -- Epoch 1/5 done -- train_loss=1.23 | train_acc=0.45 | test_loss=1.34 | test_acc=0.50 | time=180s
```

### Metrics CSV convention

```
epoch,train_loss,train_acc,test_loss,test_acc,elapsed_s,lr
1,1.234000,0.456000,1.345000,0.500000,180.0,0.089000
2,0.890000,0.650000,1.010000,0.620000,360.0,0.045000
```

One row per epoch. Write header on creation, append rows immediately after each epoch (crash-safe).

### PNG conventions

- 4-panel figure (2×2 grid), overwritten periodically (every N batches)
- Panel 1: Loss curve (raw + moving average)
- Panel 2: Accuracy / evaluation metric
- Panel 3: Learning rate schedule
- Panel 4: Loss distribution histogram (recent batches)
- Always include reference lines (baselines, best-so-far)

## File paths

The Colab VM's working directory is `/content/`. Uploaded files with relative paths land there. Use `colab ls` to verify what's on the VM.

## Related docs

- `docs/colab-gpu-keepalive.md` — Root cause: IAM deadlock, WebSocket as primary liveness signal, relay handoff protocol
- `docs/websocket-stability-china.md` — China WebSocket stability: NAT/GFW/proxy layer analysis, ping gap, mitigation
- `docs/core-flows.md` — Command-level sequence diagrams (new, exec, upload, keep-alive, relay handoff, stop)
- `docs/google-colab-cli-source-analysis.md` — Full source code architecture reference (v0.5.11)
- `docs/guides/colab-drivemount.md` — Google Drive mount: OAuth flow, automation, 120s timeout, BUSY state, gotchas (tested 2026-06-14)
- `../.colab-gpu-status.json` — Per-account GPU quota availability snapshot (T4, checked via probe)
- `../.colab-session-status.json` — Per-account active session counts (GPU/CPU, checked via `colab sessions`)
