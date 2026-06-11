# Kaggle CLI Gotchas

Field-tested patterns that differ from what you'd expect. These complement the SKILL.md quick reference with deeper context.

## 1. P100 PyTorch incompatibility (CUDA 12.8 dropped sm_60)

**Symptom:** `torch.cuda.is_available()` returns `True`, GPU is detected by `nvidia-smi`, but any actual tensor operation fails with:
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Root cause:** Kaggle's pre-installed PyTorch 2.10.0+cu128 (CUDA 12.8) dropped support for Pascal architecture GPUs (sm_60). The P100 is sm_60. T4 (sm_75) works fine. CUDA 12.8's release notes confirm Kepler, Maxwell, and Pascal were removed.

**Fix:** Force-reinstall PyTorch with CUDA 12.6 (last version supporting sm_60):

```python
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "--force-reinstall",
    "torch", "torchvision",
    "--extra-index-url", "https://download.pytorch.org/whl/cu126"
], check=True, timeout=300)
```

This takes ~3.5 minutes. On a T4 session, the reinstall is harmless overhead.

**Detect-first optimization:** Check GPU type before reinstalling to skip the 3.5 min on T4:

```python
import subprocess, sys

r = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True
)
if "P100" in r.stdout:
    print("P100 detected — reinstalling PyTorch for sm_60 support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall",
        "torch", "torchvision",
        "--extra-index-url", "https://download.pytorch.org/whl/cu126"
    ], check=True, timeout=300)
else:
    print(f"GPU: {r.stdout.strip()} — pre-installed PyTorch is compatible")

import torch
assert torch.cuda.is_available(), "CUDA not available"
```

## 2. GPU assignment is random — T4 x2 or P100

You cannot choose which GPU you get. Session-to-session, you might get a P100 (16GB, sm_60, needs PyTorch reinstall) or T4 x2 (~32GB, sm_75, works out of the box). Code defensively — handle both.

T4 x2 gives two GPU devices (`torch.cuda.device_count() == 2`) with ~16GB each. Use `DataParallel` or `DistributedDataParallel` to leverage both. P100 is a single GPU.

## 3. Push URL slug ≠ actual kernel slug (auto-conversion)

Kaggle slugifies the `title` field to create the actual kernel slug, which may differ from the `id` you specified. The push output URL shows the real slug:

```
# You set:  "id": "xieming1998/kaggle-cli-skill-test"
# Push says: https://www.kaggle.com/code/xieming1998/kaggle-cli-skill-test
# But the actual slug may differ if title doesn't match id!
```

**Always copy the slug from the push output URL**, not from your `kernel-metadata.json`. Use `kaggle kernels list --mine` to verify. Using the wrong slug returns a misleading error:

```
Permission 'kernels.get' was denied
```

This isn't actually a permissions error — it's a "slug not found" error. The kernel exists at a different slug.

**Prevention:** Make your `title` slugify to your `id`:
```
id: "xieming1998/my-training-v2"
title: "My Training v2"          # slugifies to "my-training-v2" — matches
```

## 4. Kernel slug must be unique per experiment

Pushing to an existing `id` creates a new *version* of the same kernel, not a separate run. For distinct experiments (different hyperparams, datasets), use different slugs:

```json
// Good — separate slugs for separate experiments
{"id": "xieming1998/alexnet-baseline"}
{"id": "xieming1998/alexnet-lr0.01"}
{"id": "xieming1998/alexnet-augmented"}

// Bad — same slug overwrites
{"id": "xieming1998/alexnet"}  // pushed again → version 2, not a new run
```

## 5. kernel-metadata.json: id format matters

The `id` field must match `owner/slug`. The slug part of `title` should resolve to the `id` (Kaggle warns if it doesn't). Use lowercase, hyphens, no special chars:

```
OK:   "id": "xieming1998/my-training-v2"
BAD:  "id": "xieming1998/My Training v2!"  (spaces, caps, special chars)
```

## 6. Output files are NOT saved automatically

Unlike Colab where files in `/content/` survive until the session ends, Kaggle's `/kaggle/working/` is wiped when the kernel completes. You MUST:

- Download output via `kaggle kernels output` after completion, OR
- Upload to external storage at the end of your script, OR
- Use "Save Version" in the Kaggle UI

The kernel output is available for download for a limited time after completion. Don't wait — download immediately.

## 7. Cannot stop kernels via CLI

`kaggle kernels push` starts a run, but there's no `kaggle kernels stop` command. If a run is stuck or you want to cancel, you must open the kernel URL in a browser and stop it from the Kaggle UI.

Workaround: Set conservative timeouts in your training script and save checkpoints frequently.

## 8. Token file location vs env var

The CLI reads `~/.kaggle/access_token` by default. Setting `KAGGLE_API_TOKEN` env var overrides it. For multi-account workflows:

```bash
# Per-command account switching (safe, explicit)
KAGGLE_API_TOKEN="$(cat .kaggle/access_token1)" kaggle kernels push -p ./exp1
KAGGLE_API_TOKEN="$(cat .kaggle/access_token2)" kaggle kernels push -p ./exp2

# Persistent single-account (less safe — forget which account is active)
cp .kaggle/access_token4 ~/.kaggle/access_token
```

Unlike Colab's `HOME` isolation, Kaggle CLI has no built-in multi-account support. The env var approach is recommended — it's explicit per-command and can't silently persist the wrong account.

## 9. Internet access must be enabled for pip install

Set `"enable_internet": true` in `kernel-metadata.json`. Without it, no pip install, no dataset download, no external API calls. The kernel runs in an offline sandbox.

Kaggle requires phone verification before enabling internet access on an account. Complete this before running any training that needs dependencies.

## 10. Datasets mount at /kaggle/input/<owner>/<name>

When you add `"dataset_sources": ["xieming1998/my-data"]`, the dataset appears at `/kaggle/input/xieming1998/my-data/` (read-only). The full path includes the owner namespace. Always print `os.listdir("/kaggle/input/")` first in your script to verify actual paths, since the structure can vary for competition datasets vs. your own datasets.

## 11. Script mode silently wraps your .py in a notebook

Even with `"kernel_type": "script"`, Kaggle internally converts your `.py` to a `.ipynb` notebook for execution. This means:

- `__name__` is NOT `"__main__"` — it's set by the kernel wrapper
- If you have `if __name__ == "__main__":` guards, they still work (the wrapper calls your code)
- stdout/stderr are captured per-cell rather than as a continuous stream
- `sys.exit()` raises `SystemExit` which gets logged

The wrapper is transparent for most use cases, but it explains why logs show `File "/kaggle/src/script.py"` instead of `File "train.py"`.

## 12. Pip install caching

Kaggle sessions start from a clean environment but pip packages are cached within the session. If you push multiple versions of the same kernel in quick succession, the session may be reused and packages already installed. Don't rely on this — always declare dependencies explicitly at the top of your script.

## 13. RAM varies: not always 16 GB

The docs say ~16 GB RAM, but our P100 test session showed ~31 GB. The RAM allocation varies and is not documented. Plan for 16 GB, but your script may get more. Monitor with:

```python
with open("/proc/meminfo") as f:
    for line in f:
        if "MemTotal" in line:
            print(f"RAM: {int(line.split()[1]) / 1024**2:.0f} GB")
```

## 14. GPU slot limit: 2 concurrent kernels per account

Kaggle free tier allows **2 concurrent GPU kernels** per account (not 1). Pushing a third GPU kernel while both slots are occupied raises:

```
Kernel push error: Maximum batch GPU session count of 2 reached.
```

If both slots are stuck (e.g., from kernels that you can't stop via CLI), you must manually stop them from the Kaggle website before pushing new GPU work. CPU kernels (`enable_gpu: false`) don't count toward this limit.

**Never push replacement kernels without first confirming the old one is truly dead** — see #17 on log streaming delays. Pushing duplicates when the original is actually running silently fills both slots and blocks all further work.

For parallel GPU jobs across accounts, push simultaneously with different tokens:

```bash
KAGGLE_API_TOKEN="$(cat .kaggle/access_token1)" kaggle kernels push -p ./exp1 &
KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)" kaggle kernels push -p ./exp2 &
wait
```

## 15. Python version is fixed

Kaggle currently runs Python 3.12.13 (as of June 2026). You can't choose a different Python version. Some older packages may not have Python 3.12 wheels — be prepared to install from source or use newer alternatives.

## 16. Error logs include nbconvert noise

When a kernel errors, the log output includes noise from Kaggle's notebook conversion pipeline:
```
/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning
[NbConvertApp] Converting notebook __script__.ipynb to html
[NbConvertApp] Writing 283181 bytes to __results__.html
```

This is harmless — the actual error is earlier in the log output. The conversion happens after your script fails, to generate the error page.

## 17. Log streaming can be delayed — RUNNING with no logs does NOT mean stuck

**This is the most costly Kaggle gotcha we've hit.** A kernel that shows `RUNNING` with zero log output may actually be executing normally. Kaggle's log streaming can be buffered — all 105 log lines appeared at once upon completion after 37 minutes of silence.

**Field example:** A ViT training script ran 3 experiments over 37 minutes on a P100. `kaggle kernels status` showed `RUNNING` the entire time. `kaggle kernels logs` returned 1 byte (a newline). The script was executing the whole time — GPU cycles were consumed, model checkpoints were written, metrics were logged. All logs appeared atomically at `COMPLETE`.

**What NOT to do:**
- Don't push replacement kernels assuming the original is stuck — you'll exhaust both GPU slots
- Don't ask the user to manually stop kernels based on empty logs alone
- Don't spend cycles repeatedly checking logs every 2 minutes

**What to do instead:**
1. **Check time:** If the kernel has been running less than its expected total duration, wait. Our ViT script was 37 min — checking at 5, 10, or 20 min was premature.
2. **Check for side effects:** If the script writes to datasets or external storage, check those for signs of life.
3. **Push a minimal GPU test kernel (different account)** to verify Kaggle GPU infrastructure isn't globally down.
4. **Wait for the expected duration + 50% buffer** before concluding a kernel is truly stuck.
5. **One diagnostic push only:** If you must test, push ONE minimal kernel. Don't fill both GPU slots with duplicate attempts.

**Why this happens:** Kaggle's log pipeline appears to buffer stdout/stderr for longer-running scripts, especially on GPU + internet sessions. The exact trigger is unclear (script length? output volume? session type?) but the pattern is consistent: logs appear only at completion for scripts > ~100 lines with `enable_gpu: true` + `enable_internet: true`.
