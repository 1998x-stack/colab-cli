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

### Colab official limits vs. observed behavior from China

**Official free-tier limits**: 12h max session, ~90min idle timeout. GPU quota is dynamic — heavy usage triggers 12-24h cooldown before GPU is available again.

**Observed from China**: `colab exec` frequently drops after ~8-12 min of wall-clock time (carrier-dependent). This is NOT Colab killing the session — it's the WebSocket connection dying through the SOCKS5 proxy. The session itself (and any detached training) survives, but interactive exec becomes unreachable.

The keep-alive daemon (auto-spawned by `colab new`, calls `KeepAliveAssignment` RPC via REST every 60s, max 24h) prevents idle timeout. But it uses REST API (`colab.pa.googleapis.com`), not WebSocket — so it does nothing for exec stability.

See `docs/websocket-stability-china.md` for the full root-cause analysis.

### Why exec drops: two-path architecture

Colab uses separate network paths:

| Path | Protocol | Library | Proxy support |
|------|----------|---------|--------------|
| REST (keep-alive, new, stop) | HTTPS | `requests` | Auto-detects `HTTP_PROXY`/`HTTPS_PROXY`, supports `socks5://` |
| WebSocket (exec) | WSS | `websocket-client` | Does NOT pass proxy params; `proxy_type` defaults to `"http"` |

`KernelWebSocketClient._run_websocket()` calls `run_forever()` without proxy parameters. The library reads `https_proxy` env var but defaults `proxy_type="http"`, incompatible with SOCKS5 proxies.

**Recommended config:** `HTTPS_PROXY=socks5://127.0.0.1:7890` (REST through SOCKS5) + `no_proxy="*.colab.dev,*.prod.colab.dev"` (WebSocket direct). Flip if direct fails.

### Free tier sessions can die in under 5 minutes after connection errors

After `RuntimeError: Connection was lost` or `TimeoutError` during exec, the session is often pruned on the next `colab sessions` check. Connection errors appear to accelerate termination.

**Mitigation:**
- Minimize exec failures — use detached bootstrap scripts (see Network & proxy section)
- If exec fails, immediately check with `colab sessions` and re-provision if needed
- Don't assume the session survived just because it was created 2 minutes ago

### GPU quota exhaustion

After sustained GPU usage, Colab may deny further GPU access with `TooManyAssignmentsError`. Cooldown: 12-24h for light usage, potentially days for heavy usage.

**Mitigation:**
- Switch to another account (`cb`, `cc`, `clb`) — each has independent GPU quota
- Use Kaggle Notebooks as fallback (30h/week GPU, transparent quota)
- Fix bugs locally, provision + upload + launch immediately — don't debug on the VM
- `colab run` is one-shot (provision → run → teardown) — fine for batch jobs

### Stale local session cache

The CLI caches session info locally. After a session dies on the server, the local cache may still show it. `colab sessions` refreshes from the server and prunes stale entries. The message `[colab] Pruned 1 stale local session(s).` during `colab exec` means the session died silently.

### Data caching on VM saves ~45s per re-run

The first Colab session on a project downloads data, trains tokenizers, and pre-tokenizes. These cached artifacts in `/content/` vanish when the session ends, but within a single session they survive across multiple `colab exec` calls. When re-provisioning after a session dies:

- Data download + tokenizer training + pre-tokenization: ~45s on T4
- With cached data still on `/content/` from prior exec: 0s, training starts immediately
- **Plan for it:** If you need multiple runs (e.g., hyperparameter tuning), do data prep once then re-launch training multiple times within the same session. When the session dies, the next session pays the 45s tax again.

**Observed in:** transformer-ln-comparison (2026-06-14) — second run skipped the 18MB IWSLT download and tokenization, starting training ~45s faster.

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

The Colab API (`colab.pa.googleapis.com` for REST, `*.prod.colab.dev` for WebSocket) are Google services blocked in mainland China. Route through a local proxy (Clash/Meta, mixed-port 7890).

**Two separate network paths** with different proxy behavior — see the next section for the root cause. The recommended config:

```bash
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

Without proxy env vars, all `colab` commands fail with `SSLError: UNEXPECTED_EOF_WHILE_READING`.

### Proxy + WebSocket is unstable — the two-path root cause

The kernel WebSocket (`wss://*.colab.dev`) and REST API (`colab.pa.googleapis.com`) use **different network libraries** with different proxy behavior:

- **REST** uses `requests` — reads `HTTP_PROXY`/`HTTPS_PROXY`, supports `socks5://` prefix
- **WebSocket** uses `websocket-client` — `KernelWebSocketClient._run_websocket()` calls `run_forever()` **without proxy parameters**. The library reads `https_proxy` but defaults `proxy_type="http"`, which is incompatible with SOCKS5.

This means `ALL_PROXY=socks5://...` correctly proxies `colab new`/`colab stop` but silently fails to proxy `colab exec`'s WebSocket. The WebSocket gets a raw TCP connection that may be blocked by GFW or misinterpreted by the HTTP proxy handler.

**Recommended config** — REST through SOCKS5, WebSocket direct:
```bash
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

If WebSocket direct fails, flip: remove `no_proxy`, use `HTTPS_PROXY=http://...` (WebSocket treated as HTTP CONNECT tunnel by the library).

The correct combination varies per session — flip and retry.

See `docs/websocket-stability-china.md` for the full root-cause analysis with source code references.

### SSL errors are usually transient

`SSLError: UNEXPECTED_EOF_WHILE_READING` happens occasionally. The session is often still alive — re-run the command. Only create a new session if `colab sessions` shows it's gone.

### Session URLs change

The session's backend URL (the `gpu-t4-s-kkb-...colab.dev` host) can change across kernel restarts. If `colab exec` suddenly fails with connection errors, check `colab status` to see if the session is still valid.

### REST upload doesn't need WebSocket — but exec does

`colab upload` goes through the **REST** path (HTTPS PUT to `*.prod.colab.dev`), NOT WebSocket. The upload is reliable even when exec WebSocket is down. However, `colab exec` uses WebSocket (WSS), which is the unstable path from China.

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

### `colab upload` creates a FILE (not a directory) when path doesn't exist

When uploading to `/content/myproject/script.py` and `/content/myproject/` doesn't exist, `colab upload` creates a **file** named `/content/myproject` containing the first uploaded script. All subsequent uploads to `/content/myproject/...` **overwrite** that same file — they don't create a directory.

**Symptoms:**
- First upload to `/content/s1-t4/budget_forcing.py` → creates file `/content/s1-t4` (not directory)
- Second upload to `/content/s1-t4/train.py` → overwrites `/content/s1-t4` file
- Third upload → overwrites again
- After all uploads: `/content/s1-t4` is a single file (the last uploaded script), not a directory with multiple scripts
- Any code that expects `/content/s1-t4/` to be a directory fails with `NotADirectoryError`

**Fix:** Upload flat to `/content/` root, then create directories and move files via exec. But the better fix is to **skip upload entirely** — use the base64 embed pattern below for multi-file projects.

```bash
# Upload to root (safe — /content/ always exists):
colab upload local.py /content/cot.py

# Create dir and move:
echo 'import os, shutil; os.makedirs("/content/strategies", exist_ok=True); shutil.move("/content/cot.py", "/content/strategies/cot.py")' | colab exec -s <name>
```

If this has already happened, fix on the VM:
```bash
echo 'import os; os.remove("/content/s1-t4"); os.makedirs("/content/s1-t4", exist_ok=True)' | colab exec -s <name>
```

### Multi-file deploy: use base64 embed for single-exec efficiency

`colab upload` uses REST (HTTPS PUT to `*.prod.colab.dev`), which is reliable even when WebSocket drops. But each upload is a separate REST call, and multi-file projects can be slow. For projects with many files, skip upload entirely — generate a Python script that embeds all project files as base64 and writes them to `/content/` in a single `colab exec`:

```python
# On local machine, generate the deploy script:
import os, base64

proj_dir = "projects/my-project"
lines = ['import os, base64',
         'os.makedirs("/content/my-project/logs", exist_ok=True)',
         'os.makedirs("/content/my-project/checkpoints", exist_ok=True)']

for fname in os.listdir(proj_dir):
    if fname.endswith('.py'):
        with open(os.path.join(proj_dir, fname)) as f:
            encoded = base64.b64encode(f.read().encode()).decode()
        lines.append(f'with open("/content/{fname}", "w") as f:')
        lines.append(f'    f.write(base64.b64decode("{encoded}").decode())')
        lines.append(f'print("Written: /content/{fname}")')

with open('/tmp/deploy_scripts.py', 'w') as f:
    f.write('\n'.join(lines))

# Deploy in a single exec (one WebSocket call, returns in seconds):
colab exec -s <name> -f /tmp/deploy_scripts.py --timeout 60
```

This pattern works because `colab exec -f` sends a single script for execution — it doesn't need the WebSocket to stay alive for multiple uploads. The base64 overhead is ~33%, fine for scripts totaling <100KB. For large data files (>1MB), generate them on the VM directly (download from HF, run a data prep script, etc.).

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

### fetch.sh: tar via exec can fail — always have a fallback

The `tar` step in a fetch script runs through `colab exec` (WebSocket). When the WebSocket is down or the kernel is busy, the tar operation fails silently. Always implement a fallback that downloads individual files via `colab download` (REST), which survives WebSocket drops:

```bash
# Try tar + download (fast, single file)
tar_output=$(echo '...' | colab exec -s "$session" --timeout 15 2>/dev/null) || true

if echo "$tar_output" | grep -q "TAR_OK"; then
    colab download -s "$session" /content/fetch.tar.gz "$local_dir/fetch.tar.gz"
    tar -xzf "$local_dir/fetch.tar.gz" -C "$local_dir/"
else
    # Fallback: download individual files via REST
    colab download -s "$session" /content/output/logs/train.log "$local_dir/logs/train.log" 2>/dev/null || true
    colab download -s "$session" /content/output/metrics.csv "$local_dir/metrics.csv" 2>/dev/null || true
fi
```

Also exclude checkpoints from the tar (`--exclude=checkpoints`) to keep the download under the ~624MB proxy limit. Typical metrics + logs + PNGs tar to 50-80KB.

**Observed in:** transformer-ln-comparison fetch.sh (2026-06-14)

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

## Drive mount

### `colab drivemount` internal timeout is 120s, not the CLI's 600s

The CLI has a 600s timeout (`automation.py:35`), but `drive.mount()` on the VM has an internal 120s timeout (`timeout_ms=120000`). The critical chain:
1. CLI prints OAuth URL → user completes browser auth → user presses Enter
2. CLI calls `credentials-propagation` API
3. Kernel receives `input_reply` and resumes `drive.mount()`
4. `drive.mount()` waits up to 120s for DFS credentials to be ready
5. If credentials not ready: `ValueError: mount failed`

The 120s clock starts when the kernel receives the `input_reply`, NOT when the URL is printed. But the practical constraint is: **complete browser OAuth within ~90s** of the URL appearing, to leave time for propagation.

**Mitigation:** Open the browser immediately. If you miss the window, `colab stop` + `colab new` + retry.

### Killed drivemount leaves session in unrecoverable BUSY state

If `colab drivemount` is killed (Ctrl+C, SIGTERM, network drop), the kernel is still executing `drive.mount()` and waiting for a WebSocket `input_reply` that will never arrive. Session status shows `BUSY (automation(drivemount))`.

**You cannot re-run drivemount on this session** — the kernel is stuck in the mount flow. Must `colab stop -s <name>` + `colab new -s <name>` to get a fresh kernel.

### Drive mount works on CPU sessions (no GPU needed)

`colab drivemount` only needs a running session — any variant works. Use `colab new -s <name>` (CPU) to avoid consuming your one free GPU slot. This is useful for:
- Checking what's on Drive before deciding whether to provision GPU
- Downloading data from Drive to local, then uploading to a GPU session
- Verifying that Drive auth is cached before GPU provisioning

### First mount per account requires browser OAuth; subsequent mounts auto-succeed

The first time an account uses `colab drivemount`, you must complete browser OAuth. After that, the OAuth token is cached by Colab's backend — `credentials-propagation` succeeds without prompting. To pre-authorize without CLI: open a browser Colab notebook, click "Mount Drive", and complete the auth flow there. The authorization is per-account, not per-session.

### The `input()` prompt has no trailing newline — `readline()` blocks forever

When automating drivemount with `subprocess.Popen` and `proc.stdout.readline()`, the reader blocks after the URL line because the Colab kernel's `input("Press Enter...")` prints the prompt WITHOUT a trailing newline and then waits for stdin. `readline()` waits for `\n` which never arrives.

**Fix for automation:** Use a background thread to read stdout with `iter(proc.stdout.readline, "")` while the main thread scans accumulated output via regex for the OAuth URL. The thread blocks harmlessly; the main thread proceeds as soon as the URL is detected.

## Authentication

### Multi-account: tokens expire independently, `whoami` ≠ `sessions`

Each account alias (`cb`, `cc`) has its own OAuth token in `~/colab-accounts/account-{b,c}/.config/colab-cli/token.json`. Tokens need periodic refresh:

- `colab whoami` triggers an OAuth token refresh (hits `oauth2.googleapis.com`)
- `colab sessions` uses cached session data — can succeed even when `whoami` fails

If `whoami` fails with "failed to refresh credentials" but `sessions` works: the token refresh endpoint is having proxy issues. The sessions data comes from local cache and may be stale. Try `whoami` again — the account may need re-authentication.

### OAuth flow

First run opens a browser URL for Google OAuth. Complete it in your browser. The token is cached at `~/.colab-cli-oauth-config.json`.

If auth fails, the CLI prints the URL again. No explicit `colab login` command exists — just run any session command and auth will trigger automatically.

---

## Large-scale benchmark deployment (2026-06-14)

### GPU quota exhausts across ALL accounts simultaneously

When running 19+ GPU experiments across a single session, all 4 free-tier accounts exhausted GPU quota (412 TooManyAssignmentsError). The 12-24h cooldown applies to ALL accounts, not per-account. Lesson: spread GPU usage across days, not hours. For mass experiments, use Kaggle (30h/week GPU, more reliable).

### Quick benchmarks should run inline, not detached

For benchmarks that complete in <3 minutes, running inline within a single `colab exec --timeout <N>` is safer than launching detached subprocesses and polling. The inline exec holds the WebSocket open, providing live output. Detached subprocesses may complete but their output files vanish when the session dies before the next cron fetch.

```bash
# Better for <3 min benchmarks:
cat <<'PYEOF' | colab exec --timeout 240
import subprocess, sys
proc = subprocess.Popen([sys.executable, "-u", "/content/benchmark.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
stdout, _ = proc.communicate(timeout=220)
print(stdout.decode())
PYEOF
```

### First Colab session is lost to data download overhead

CIFAR-10 download (~170MB) + CUDA JIT = 7-10 min overhead. Combined with ~10 min GPU window, the first session rarely completes training. Use a short warmup session to cache data at `/content/data/`, then re-provision.

### Config B (HTTP CONNECT) is more reliable for full-session workflows

When running upload→exec→download chains, Config B (`HTTPS_PROXY=http://`, `ALL_PROXY=socks5://`) was consistently more reliable than Config A (`HTTPS_PROXY=socks5://` + `no_proxy`). Config A's no_proxy exclusion for `*.prod.colab.dev` causes SSL/EOF errors on upload/download REST calls. When in doubt, start with Config B.

### Session lives 30+ minutes with active WebSocket

Despite the documented ~10 min window, one session survived 30+ minutes with continuous detached benchmark runs and periodic `colab exec` checks. The key: keep the WebSocket active every few minutes. Even a quick `echo 'print("alive")' | colab exec` resets the ~2-3 min post-WebSocket death timer.

### Cron watchtower for detached benchmarks

For benchmarks lasting 3+ minutes, set a cron watchtower that fetches outputs every 2 minutes via REST (`colab download`). This survives individual WebSocket drops:

```bash
CronCreate(
    cron="*/2 * * * *",
    prompt="Fetch results from Colab: 1. tar /content/cuda-dark-corners-output/ 2. colab download tar 3. report tail. Skip if 404/401.",
    recurring=True,
)
```

Cancel the cron when benchmarks complete to avoid noise.
