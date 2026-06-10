# Background Training on Colab GPU

You have `train_cifar.py` at `~/ml/train_cifar.py`. It needs `torch`, `torchvision`, `wandb`, saves checkpoints to `./checkpoints/` every 10 epochs, and takes ~3 hours.

## Full setup

### 1. Provision a GPU VM

```bash
colab new --gpu T4 -s training
```

T4 is the most reliable GPU on free tier. If you have Colab Pro/Pro+, you can try `--gpu L4` or `--gpu H100`.

### 2. Upload your training script

The VM's working directory is `/content/`. Upload your script there:

```bash
colab upload ~/ml/train_cifar.py /content/train_cifar.py
```

Your script already writes checkpoints to `./checkpoints/`, which resolves to `/content/checkpoints/` on the VM -- no change needed.

### 3. Create a launch proxy script

This is the critical piece. A `colab exec` call times out after 30+ seconds. You need a lightweight launcher that pip-installs dependencies and spawns your training script as a detached subprocess that survives after `colab exec` returns.

Create `launch.py` (locally, wherever you run `colab exec -f` from):

```python
import subprocess, sys, os

DEPS = ["torch", "torchvision", "wandb"]
SCRIPT = "/content/train_cifar.py"
LOG = "/content/train.log"

subprocess.check_call([sys.executable, "-m", "pip", "install"] + DEPS + ["-q", "--progress-bar", "off"])

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
print(f"Training launched. PID={proc.pid} log={LOG}")
```

Three things keep the process alive after `colab exec` disconnects:

- **`start_new_session=True`** -- creates a new process group so the child doesn't receive SIGHUP when the kernel WebSocket drops.
- **`PYTHONUNBUFFERED=1` + `python -u`** -- forces unbuffered stdout so the log file populates immediately instead of sitting empty for hours.
- **Pip installs happen inline before spawning** -- the training process only starts once everything is ready.

### 4. Launch the training (detached)

```bash
colab exec -f launch.py --timeout 120
```

This sends `launch.py` to the VM, installs torch/torchvision/wandb (2-3 minutes), spawns `train_cifar.py` as a detached subprocess, then returns. The training runs in the background regardless of what happens to your laptop (sleep, lid closed, network drop).

### 5. Create a check-progress script

Create `check_progress.py` (locally):

```python
import subprocess, sys

LOG = "/content/train.log"
CKPT_DIR = "/content/checkpoints"
SCRIPT_NAME = "train_cifar.py"

# Check if training process is alive
result = subprocess.run(
    ["pgrep", "-f", SCRIPT_NAME],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"Process alive: PID(s)={result.stdout.strip()}")
else:
    print("Process NOT running (may have finished or crashed)")

# Tail last 20 lines of log
print(f"\n--- Last 20 lines of {LOG} ---")
subprocess.run(["tail", "-20", LOG])

# List checkpoint files
print(f"\n--- Checkpoints in {CKPT_DIR} ---")
subprocess.run(["ls", "-lh", CKPT_DIR])
```

Check progress at any time:

```bash
colab exec -f check_progress.py --timeout 15
```

### 6. Download checkpoints mid-run

Don't wait until the session dies to retrieve your work. Download checkpoints during training:

```bash
# Tar the checkpoints directory (colab download doesn't do directories)
echo 'import subprocess, shutil, os; subprocess.run(["tar", "-czf", "/content/checkpoints.tar.gz", "-C", "/content", "checkpoints"])' | colab exec --timeout 30
colab download /content/checkpoints.tar.gz ./checkpoints_epoch50.tar.gz
```

### 7. Stop the session when done

```bash
colab stop -s training
```

## Important caveats

- **3-hour job is tight on free tier.** Free GPU sessions auto-terminate after ~2-4 hours total. Your 3-hour training could be cut off. If possible, enable resumption from the latest checkpoint in your script -- that way if the session dies, you can provision a fresh one and resume. If you have Colab Pro/Pro+, this is less of a concern.
- **Checkpoints directory.** Your script writes to `./checkpoints/`, which is `/content/checkpoints/` on the VM. The first time you check progress, the directory may not exist yet (early epochs). This is fine.
- **`colab exec -f` reads local files.** The `-f` flag sends `launch.py` and `check_progress.py` from your current working directory to the VM. It does not run files already on the VM.
- **Log file.** The training log goes to `/content/train.log`. If you ever see an empty log, check that `PYTHONUNBUFFERED=1` and `-u` are both set.
- **Session health.** If a `colab exec` command fails with an SSL/connection error, the session may still be alive. Run `colab sessions && colab status` to check.
- **Wandb.** If your script uses `wandb.init()`, the Colab VM has direct internet access from GCP -- no proxy needed. The first run will prompt for a wandb API key. Either set `WANDB_API_KEY` in the launch env, or your code can pick it up from `~/.netrc` on the VM.

## Alternative: one-shot run

If you don't need to monitor progress interactively and your script handles everything (checkpoints to a downloadable location, wandb logging), you can use the simpler one-shot approach:

```bash
colab run --gpu T4 launch.py
```

This provisions a VM, runs the script, and auto-teardowns -- no session management, no stopping. But you can't check progress mid-run or download partial results.
