# Running ~/ml/train_cifar.py on Colab GPU (Background / 3-hour Training)

Below is the full setup based on the `colab-cli` skill. The key requirement is that the training process **survives `colab exec` disconnection** so it keeps running if your laptop goes to sleep. That means using `subprocess.Popen` with `start_new_session=True` in a launcher script.

> **Important:** Free-tier Colab sessions last ~2-4 hours. Your 3-hour training is feasible but tight -- checkpoints are essential insurance. Prefer a T4 GPU, which is the most reliable free-tier accelerator.

---

## Step 1: Provision a GPU VM

```bash
colab new --gpu T4 -s training
```

This creates a Colab session named `training` with a Tesla T4 (16 GB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128).

---

## Step 2: Upload `train_cifar.py`

```bash
colab upload ~/ml/train_cifar.py /content/train_cifar.py
```

**Always use absolute remote paths** (`/content/...`). Relative remote paths can silently fail.

---

## Step 3: Create a launcher script (`launch.py`)

This script runs **locally** (you `colab exec -f` it). It pip-installs dependencies on the VM, then spawns your training as a detached subprocess that survives exec disconnection.

Create this file in your local project directory:

```python
# launch.py
import subprocess, sys, os, time

SCRIPT = "/content/train_cifar.py"
DEPS = ["torch", "torchvision", "wandb"]
LOG = "/content/train.log"

# -- Install dependencies --
for pkg in DEPS:
    print(f"[launch] pip install {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# -- Spawn training (detached, unbuffered) --
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,   # survives exec disconnect
        env=env,
    )

time.sleep(3)
if proc.poll() is not None:
    print(f"[launch] ERROR: exited immediately (code={proc.returncode})")
    subprocess.run(["tail", "-20", LOG])
    sys.exit(1)

print(f"[launch] OK. PID={proc.pid}  log={LOG}")
```

**Key details:**
- `start_new_session=True` detaches the child into its own process group so it does NOT receive SIGHUP when `colab exec` disconnects.
- `PYTHONUNBUFFERED=1` + `python -u` force unbuffered stdout so log output appears in real time.
- **`colab exec -f` only takes a relative path** (relative to your local CWD). Keep this file in the directory you'll run from.

---

## Step 4: Ensure `train_cifar.py` creates the checkpoints directory

Your script saves to `./checkpoints/`. On the Colab VM, CWD is `/content/`, so checkpoints go to `/content/checkpoints/`. Make sure your script creates this directory:

```python
os.makedirs("checkpoints", exist_ok=True)
```

---

## Step 5: Launch the training

```bash
cd /path/where/launch.py/is
colab exec -f launch.py --timeout 120
```

This sends `launch.py` to the VM for execution. The launcher installs `torch`, `torchvision`, `wandb`, then spawns your training in the background and exits. The whole `colab exec` call should complete in well under 120 seconds.

After this, the training is running detached. You can close your laptop -- it continues.

---

## Step 6: Check progress

Create a simple check script (`check_progress.py`) or use one-liners:

```bash
# Check if process is alive
echo 'import subprocess; print(subprocess.run(["pgrep", "-f", "train_cifar"], capture_output=True, text=True).stdout)' | colab exec --timeout 10

# Tail the log
echo 'with open("/content/train.log") as f: lines = f.readlines(); print("".join(lines[-15:]))' | colab exec --timeout 10

# Check checkpoints
echo 'import os; print(os.listdir("/content/checkpoints")) if os.path.isdir("/content/checkpoints") else print("no checkpoints dir")' | colab exec --timeout 10
```

Or upload and exec a `check_progress.py` script (see the skill's `scripts/check_progress.py` for a template that checks process + log + checkpoints in one shot).

---

## Step 7: Download checkpoints

```bash
colab download /content/checkpoints/epoch_10.pt ./checkpoints/
colab download /content/checkpoints/epoch_20.pt ./checkpoints/
# ... etc, one file at a time
```

`colab download` only handles individual files. For bulk download, tar the directory on the VM first:

```bash
echo 'import subprocess; subprocess.run(["tar", "-czf", "/content/checkpoints.tar.gz", "-C", "/content", "checkpoints"])' | colab exec --timeout 15
colab download /content/checkpoints.tar.gz ~/ml/checkpoints.tar.gz
```

---

## Step 8: Stop the session when done

```bash
colab stop -s training
```

This releases the VM. Idle VMs burn compute units even when not actively training.

---

## Critical gotchas (from the skill)

| Gotcha | Detail |
|--------|--------|
| **Session lifetime** | Free-tier sessions last ~2-4 hours total. Your 3-hour training is borderline. Download checkpoints periodically. |
| **`colab exec -f` uses relative local paths** | Must be a path relative to your local CWD. Absolute paths fail. |
| **`colab exec` has no `-c` flag** | Use `echo 'code' \| colab exec` for inline one-liners. |
| **`start_new_session=True` is required** | Without it, the child process dies when exec disconnects. |
| **`PYTHONUNBUFFERED=1` is required** | Without it, log files appear empty despite active training. |
| **Upload uses absolute remote paths** | Always use `/content/filename.py` for the remote path. |
| **`colab download` does not support directories** | Tar first on the VM, then download the tarball. |
| **Proxy required from China** | Set `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` before every command. If `colab exec` gets `Connection was lost`, add `no_proxy="*.colab.dev"`. |
| **Only 1 GPU session per free account** | Use multi-account aliases (`cb`, `cc`) for parallel GPU sessions. |

---

## If training exceeds the session window

3 hours is near the free-tier limit. Mitigations:

1. **Checkpoint resume:** Make sure `train_cifar.py` can resume from the latest checkpoint. That way, if the session dies at 2.5 hours, you can create a new one and restart from where it left off.
2. **Upgrade to Colab Pro/Pro+** for longer sessions (L4, H100, A100 available).
3. **Use `colab run`** for truly ephemeral jobs (auto-provision + execute + teardown), though this sacrifices monitoring.
