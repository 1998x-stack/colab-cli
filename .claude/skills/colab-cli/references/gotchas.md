# Colab CLI Gotchas

Field-tested surprises that differ from expected behavior.

## Path handling

### `colab exec -f` reads LOCAL files, NOT remote VM files

The `-f` flag reads the Python file from your LOCAL filesystem and sends it to the VM. The file path is relative to your local CWD, NOT relative to `/content/` on the VM.

This has two critical implications:

**1. Local CWD matters for `-f`:**

```bash
# You're in ~/projects/my-project/ which has launch.py
colab exec -f launch.py    # sends ~/projects/my-project/launch.py ✓

# Wrong directory = wrong file
cd ~/other-project
colab exec -f launch.py    # sends ~/other-project/launch.py ✗
```

Always `cd` to the project directory before `colab exec -f`, or use an absolute local path.

**2. Upload is only needed for subprocess scripts:**

`-f` sends the file to the kernel for execution. But if that script spawns subprocesses referencing `/content/worker.py`, those files must be uploaded separately:

```bash
# launch.py spawns: subprocess.Popen(["python", "/content/compare.py"], ...)
# So compare.py must be on the VM — upload it:
colab upload compare.py /content/compare.py

# But launch.py itself is sent via -f — no upload needed:
colab exec -f launch.py
```

### `colab exec -f` works with relative paths only (relative to local CWD)

The `-f` flag takes a path relative to your local CWD. Absolute paths fail with `FileNotFoundError`.

**Wrong:**
```bash
colab exec -f /Users/me/project/script.py    # FileNotFoundError
```

**Right:**
```bash
cd /Users/me/project
colab exec -f script.py                      # works
```

Note: newer versions of the colab CLI may support absolute paths — if relative paths are inconvenient, try absolute and check.

### Upload with relative path lands in /content

```bash
colab upload local.py remote.py     # → /content/remote.py
colab upload local.py /content/x.py # → /content/x.py (same result)
```

## Accelerator availability

### TPU is unreliable on free tier

- `--tpu v6e1`: almost always "Backend rejected accelerator" on free accounts
- `--tpu v5e1`: often returns "Service Unavailable" (503)
- T4 GPU (`--gpu T4`): most reliable free-tier accelerator
- CPU (omit both flags): always works

Always have a fallback. Try T4 first for ML workloads.

### Free tier: only 1 GPU session at a time

Provisioning a second GPU session raises `TooManyAssignmentsError (412 Precondition Failed)`. To run two GPU projects, run them sequentially: create → train → download → stop, then repeat for the second. CPU sessions are not rate-limited this way — you can have 1 GPU + 1 CPU concurrently.

## Session lifetime

### Sessions auto-terminate

Free-tier GPU (T4) sessions last **12-15 minutes** of wall-clock time. Not 2-4 hours — that's the best case for CPU or Pro. GPU sessions are aggressively terminated. There is no warning. All files are lost.

**Mitigation:**
- Keep the full pipeline under 10 minutes: pip install (60-90s) + downloads + experiment
- Pre-download data and models locally when possible, upload to VM
- Download artifacts immediately after generation
- For anything longer, split across multiple sessions or upgrade to Colab Pro
- `colab run` is one-shot (provision → run → teardown) — fine for batch jobs, but results must be uploaded to Drive or external storage from within the script since the session is destroyed after completion

### Free tier sessions can die in under 5 minutes after connection errors

After `RuntimeError: Connection was lost` or `TimeoutError` during exec, the session is often pruned on the next `colab sessions` check — much faster than the typical 2-4 hour window. Connection errors appear to accelerate termination.

**Mitigation:**
- Minimize exec failures — use detached bootstrap scripts (see Network & proxy section)
- If exec fails, immediately check with `colab sessions` and re-provision if needed
- Don't assume the session survived just because it was created 2 minutes ago

### Free tier GPU sessions die in 12-15 minutes

The 2-4 hour window is for CPU/Pro, not free GPU. Free-tier T4 GPU sessions **consistently die in 12-15 minutes** of wall-clock time, even mid-execution with no errors. This is the norm, not an edge case. From 7 provisioning attempts in one session, 4 sessions died before completing a ~10 min pipeline.

Idle time during debugging burns the clock just as fast as active compute.

**Mitigation:** Fix bugs locally, then provision a fresh session and upload+launch immediately. Don't spend time iterating on bug fixes after provisioning — the clock is ticking.

### Stale local session cache

The CLI caches session info locally. After a session dies on the server, the local cache may still show it. `colab sessions` refreshes from the server and prunes stale entries. The message `[colab] Pruned 1 stale local session(s).` during `colab exec` means the session died silently.

## Subprocess behavior

### stdout is fully buffered in subprocess

When spawning background jobs via `subprocess.Popen`, Python's default buffering means output won't appear in log files until the buffer fills (typically 8KB). For real-time log output:

```python
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.Popen(
    [sys.executable, "-u", "script.py"],  # -u = unbuffered
    stdout=f, stderr=subprocess.STDOUT,
    start_new_session=True,               # detach from parent
    env=env,
)
```

Without this, `check_progress.py` will show an empty log file even though the process is actively training.

### start_new_session=True for true nohup

Without `start_new_session=True`, the child process is in the same process group as the Colab kernel. When the kernel's execution slot ends (after `colab exec` timeout), the child may receive SIGHUP and die.

With `start_new_session=True`, the child gets its own process group and session, surviving kernel restarts and exec timeouts.

## Network & proxy

### Proxy is required from China

The Colab API (`colab.research.google.com`) and kernel proxy (`*.colab.dev`) are Google services blocked in mainland China. Route through a local proxy (Clash/Meta, mixed-port 7890) using env vars:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

`HTTPS_PROXY`/`HTTP_PROXY` covers Python `requests` (control-plane API calls). `ALL_PROXY=socks5://` is needed for WebSocket connections (kernel runtime). Without these, all `colab` commands fail with `SSLError: UNEXPECTED_EOF_WHILE_READING`.

### Proxy + WebSocket is unstable — try both with and without no_proxy

The kernel WebSocket (`wss://*.colab.dev`) sometimes breaks through the SOCKS5 proxy. If `colab exec` fails with `RuntimeError: Connection was lost`, try again with `no_proxy` set:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890
# Try 1: all through proxy
colab exec -s <name> ...

# Try 2: colab.dev direct (if Try 1 gets "Connection was lost")
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
colab exec -s <name> ...
```

The correct combination varies per session — just flip and retry.

### SSL errors are usually transient

`SSLError: UNEXPECTED_EOF_WHILE_READING` happens occasionally. The session is often still alive — re-run the command. Only create a new session if `colab sessions` shows it's gone.

### Session URLs change

The session's backend URL (the `gpu-t4-s-kkb-...colab.dev` host) can change across kernel restarts. If `colab exec` suddenly fails with connection errors, check `colab status` to see if the session is still valid.

### WebSocket drops during sustained exec — use detached bootstrap for long workflows

`colab exec` uses a Jupyter kernel WebSocket that frequently drops during sustained operations like `pip install` (30-60s+), health-check polling loops, or long downloads. Two failure modes:

- `RuntimeError: Connection was lost` — WebSocket died mid-execution, work is killed
- `TimeoutError: Timeout waiting for reply` — exec finished but reply never arrived (background work may survive)

**The fix: delegate all heavy work to a detached bootstrap script:**

```python
# launch.py — exec THIS via colab exec -f launch.py (returns in <5s)
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/bootstrap.py"],
    start_new_session=True,
)
print(f"OK. Bootstrap PID={proc.pid}")

# bootstrap.py — does pip install + worker spawn (runs detached, survives everything)
# This file must be uploaded to /content/bootstrap.py beforehand
```

The key: `launch.py` spawns `bootstrap.py` via `subprocess.Popen(start_new_session=True)` and exits immediately. The exec returns in seconds. Meanwhile, `bootstrap.py` handles all heavy work (pip install, model downloads, worker spawn) in a process group that survives WebSocket disconnection.

**Without this pattern:** pip install inside `colab exec` → WebSocket drops after 30-60s → `RuntimeError: Connection was lost` → session often pruned.

## GPU / CUDA

### CUDA version mismatch: Colab T4 has CUDA 12.8, latest vLLM wants CUDA 13

Colab T4 VMs (as of 2026-06) ship with:
- CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.12
- GPU: Tesla T4 (16 GB VRAM)

The latest vLLM default wheel requires CUDA 13 and fails with:
```
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**Fix: Pin vLLM version with `--extra-index-url` (NOT `--index-url`):**

```bash
# WRONG — installs CUDA 13 wheel from PyPI
pip install vllm

# ALSO WRONG — --index-url replaces PyPI entirely, so pip can't find
# blake3, prometheus-fastapi-instrumentator, and other non-PyTorch deps
pip install vllm --index-url https://download.pytorch.org/whl/cu128

# RIGHT — pin version, use --extra-index-url so PyPI still provides deps,
# but cu128 index wins the vLLM wheel itself
pip install "vllm>=0.10,<0.11" --extra-index-url https://download.pytorch.org/whl/cu128
```

**Why `--index-url` fails:** The cu128 index only has PyTorch-ecosystem packages. Replacing PyPI means pip can't resolve `blake3`, `prometheus-fastapi-instrumentator`, and other general deps — each fails with "Could not find a version."

**Why pinning is needed:** Without a version cap, PyPI's vLLM 0.11.x (CUDA 13) takes priority over cu128's 0.10.x (CUDA 12.8) even with `--extra-index-url`. The `>=0.10,<0.11` constraint forces pip to pick the cu128 wheel.

### vLLM + transformers version compatibility

vLLM 0.8.x is incompatible with the `transformers` version pre-installed on Colab T4 VMs:
```
ModuleNotFoundError: Could not import module 'ProcessorMixin'
```

vLLM 0.9.x installs but fails at runtime with Qwen-family tokenizers:
```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```
This is a breaking change in newer `transformers` that vLLM 0.9.x doesn't handle.

**Use vLLM >= 0.10, < 0.11** — both CUDA 12.8 compatible AND works with Colab's transformers + Qwen tokenizers.

### `pgrep` false positives in Colab VMs

`pgrep -f "server"` and `pgrep -f "compare"` often match kernel threads and system processes with those substrings in their command line. PIDs like 111, 113, 117 are NOT your processes. Always verify with a more specific pattern:

```python
# WRONG — matches kernel threads with "server" in cmdline
subprocess.run(["pgrep", "-f", "server.py"], ...)

# BETTER — include the .py extension to avoid false matches
subprocess.run(["pgrep", "-f", "server\\.py"], ...)
```

When in doubt, cross-check with `subprocess.run(["ps", "-p", pid])` to see the actual process name.

### PyTorch API: `total_memory` not `total_mem`

`torch.cuda.get_device_properties(0).total_mem` was renamed to `total_memory` in PyTorch 2.11+. Colab VMs use PyTorch 2.11.0, so use:

```python
vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
```

### numba.cuda: `get_current_device().name` type varies by version

Older numba returns `bytes` (needs `.decode()`), newer numba (≥0.61) returns `str` directly. Use a helper:

```python
def get_device_name():
    name = cuda.get_current_device().name
    return name.decode() if isinstance(name, bytes) else name
```

Colab VMs currently install numba ≥0.61 which returns `str` — calling `.decode()` on a str raises `AttributeError`.

### numba.cuda: `cuda.grid(2)` returns `(x, y)` = `(col, row)`

In CUDA's 2D grid convention, `x` maps to the innermost dimension (columns), `y` maps to the outer dimension (rows). Destructure accordingly in all 2D kernels:

```python
# CORRECT: x=col, y=row
col, row = cuda.grid(2)

# BUG: swaps row/col — causes wrong output (constant-value patterns)
row, col = cuda.grid(2)
```

This affects matmul, convolution, and any kernel launched with a 2D grid. Symptoms: output contains repeated values, max_diff is enormous.

### numba.cuda: pass device arrays explicitly with `cuda.to_device()`

When a host numpy array is passed directly to a CUDA kernel, numba auto-copies it per launch — slow and can cause correctness issues in reduction/accumulation kernels. Transfer to device explicitly:

```python
data_dev = cuda.to_device(data_host)
out_dev = cuda.to_device(out_host)
kernel[blocks, threads](data_dev, out_dev, N)
result = out_dev.copy_to_host()
```

This eliminates the `NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead` warning. Essential for reduction kernels where float32 precision across many partial sums matters.

### numba.cuda: avoid `cuda.declare_device()` — API changes across versions

The signature format for `cuda.declare_device()` varies across numba versions (`np.float32` vs `'float32[:]'` vs tuple signatures). Prefer passing arrays as regular kernel parameters instead of using constant memory. For small read-only data like convolution filters, the L1 cache is often sufficient.

### T4 GPU for CUDA learning

Colab T4 is a Turing GPU (compute capability 7.5). Detected as `Tesla T4` by numba.cuda. Good for learning CUDA programming concepts; lacks tensor cores accessible via numba (numba doesn't expose warp-level MMA). For production deep learning, Colab Pro/Pro+ with A100/H100 is better.

## HuggingFace library compatibility

### datasets >= 4.0 breaks older HF datasets

Colab VMs ship `datasets==4.0.0` which changed dataset path resolution. Many older datasets (hotpot_qa, squad, etc.) fail:

```
HfUriError: Repository id must be 'namespace/name', got 'hotpot_qa'
```

**Fix:** Pre-download data locally and upload the JSON file. Don't use `datasets` on the VM.

```bash
# On local machine:
python -c "
from datasets import load_dataset
import json
ds = load_dataset('hotpot_qa', 'distractor', split='validation')
# sample and save to JSON
"

# Upload to VM:
colab upload data.json /content/data.json
```

If you must use datasets on Colab, install `datasets==2.21.0` + `huggingface-hub==0.23.0`. But this creates version conflicts with anything needing a newer `huggingface-hub` (transformers, vLLM).

### AWQ model loading needs gptqmodel on Colab

Colab's `transformers` version requires `gptqmodel` for AWQ-quantized models, not `autoawq` alone:

```
ImportError: Loading an AWQ quantized model requires gptqmodel.
Please install it with 'pip install gptqmodel'
```

**Fix:** `pip install autoawq gptqmodel` — install both. `gptqmodel` handles AWQ model loading in newer transformers; `autoawq` provides the kernels.

### vLLM version vs. transformers compatibility on Colab

Colab T4 VMs ship specific `transformers` versions that create narrow vLLM compatibility windows:

| vLLM | Issue |
|------|-------|
| ≥0.8.0 | CUDA 13 only — `libcudart.so.13: cannot open shared object file` |
| 0.7.x | `ModuleNotFoundError: Could not import module 'ProcessorMixin'` — transformers too new |
| 0.9.x | `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` |

For Colab T4, the pragmatic options are: (a) use `vllm >=0.10, <0.11` with `--extra-index-url https://download.pytorch.org/whl/cu128`, or (b) use `transformers` + `AutoModelForCausalLM` directly (no vLLM). Option (b) avoids all CUDA/version conflicts at ~3-5s per inference (sequential, no batching).

## File management

### `colab upload` can't create subdirectories

Uploading to `/content/strategies/cot.py` when `strategies/` doesn't exist on the VM returns HTTP 500. `colab upload` does not auto-create parent directories on the VM.

**Fix:** Upload flat to `/content/` root, then create directories and move files via exec:

```bash
# Upload to root:
colab upload local.py /content/cot.py

# Create dir and move:
echo 'import os, shutil; os.makedirs("/content/strategies", exist_ok=True); shutil.move("/content/cot.py", "/content/strategies/cot.py")' | colab exec -s <name>
```

Or use a monolithic script that writes all files inline — no uploads needed for multi-file projects.

### colab ls is your debug tool

When files aren't where you expect:
```bash
colab ls                          # list files in /content
echo 'import os; print(os.listdir("/content"))' | colab exec
```

### Upload overwrites silently

`colab upload` overwrites the remote file without confirmation. The upload message says "Uploaded" regardless of whether the file was new or replaced.

### Upload: always use absolute remote paths

Relative remote paths may fail silently (upload reports success but file isn't there). Always use `/content/...` for the remote path:

```bash
# Wrong — may silently not appear
colab upload local.py train.py

# Right
colab upload local.py /content/train.py
```

### Directory download not supported

`colab download` only handles single files. To download a directory, tar it on the VM first:

```bash
echo 'import subprocess; subprocess.run(["tar","-czf","/content/results.tar.gz","-C","/content","output-dir"])' | colab exec -s <name>
colab download -s <name> /content/results.tar.gz ./results.tar.gz
```

### quoting in colab exec heredocs

Shell heredocs (`<<'EOF'`) with Python code that contains curly braces (f-strings, dicts) work fine. But avoid mixing `'` and `"` inside `echo '...'` pipes — use heredocs for multi-line Python, never `echo`.

### `colab exec` has NO `-c` flag

Unlike `python -c`, `colab exec` does not support a `-c` flag for inline code:

```bash
# WRONG — fails with "No such option: -c"
colab exec -c "print('hello')" --timeout 10

# RIGHT — pipe via stdin
echo "print('hello')" | colab exec --timeout 10
```

### Inline Python via echo is fragile with f-strings and special chars

`echo '...' | colab exec` breaks on f-strings, dicts, and backslashes because the shell interprets `$`, `\\`, and `{}` before Python sees them:

```bash
# WRONG — f-string braces break shell parsing
echo 'print(f"Running: {bool(x)}")' | colab exec
# SyntaxError: unterminated f-string literal

# RIGHT — use a separate check script and upload + exec:
colab upload check_progress.py /content/check_progress.py
colab exec -f check_progress.py
```

If you must inline, use only simple expressions — no f-strings, no dicts, no backslash escapes. Multi-line Python via `echo` is almost always wrong.

## Authentication

### Multi-account: tokens expire independently, `whoami` ≠ `sessions`

Each account alias (`cb`, `cc`) has its own OAuth token in `~/colab-accounts/account-{b,c}/.config/colab-cli/token.json`. Tokens need periodic refresh:

- `colab whoami` triggers an OAuth token refresh (hits `oauth2.googleapis.com`)
- `colab sessions` uses cached session data — can succeed even when `whoami` fails

If `whoami` fails with "failed to refresh credentials" but `sessions` works: the token refresh endpoint is having proxy issues. The sessions data comes from local cache and may be stale. Try `whoami` again — the account may need re-authentication.

### OAuth flow

First run opens a browser URL for Google OAuth. Complete it in your browser. The token is cached at `~/.colab-cli-oauth-config.json`.

If auth fails, the CLI prints the URL again. No explicit `colab login` command exists — just run any session command and auth will trigger automatically.
