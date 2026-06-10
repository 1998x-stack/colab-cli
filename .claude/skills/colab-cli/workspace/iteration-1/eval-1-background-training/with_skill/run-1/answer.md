# Running train_cifar.py on Colab GPU (background, survives sleep)

## Overview

Provision a T4 GPU session, upload your script and the two launcher helpers, and launch the training as a detached nohup-style subprocess. Because the process is in its own session group, it keeps running after `colab exec` returns -- your laptop can go to sleep, the Colab VM stays alive.

Estimated runtime: ~3 hours. Session lifetime is ~2-4 hours on free tier, so this is tight. Download checkpoints before the session auto-terminates.

---

## Step-by-step

### 1. Customize `launch.py` for your script

Copy the template from `scripts/launch.py` and change these variables:

```python
SCRIPT = "/content/train_cifar.py"
DEPS = ["torch", "torchvision", "wandb"]
LOG = "/content/train.log"
```

This installs the three dependencies, then spawns your script as a detached subprocess with unbuffered stdout (`-u` flag + `PYTHONUNBUFFERED=1`). Without unbuffered mode the log file stays empty until the buffer fills (8KB).

### 2. Customize `check_progress.py`

Copy the template from `scripts/check_progress.py` and change these variables:

```python
SCRIPT_NAME = "train_cifar"
LOG = "/content/train.log"
CHECKPOINT_DIR = "/content/checkpoints"
```

### 3. Provision and upload

```bash
# Create a T4 GPU session
colab new --gpu T4 --session cifar-run

# Upload your training script and the two helpers
colab upload ~/ml/train_cifar.py train_cifar.py
colab upload launch.py launch.py
colab upload check_progress.py check_progress.py

# Verify files landed
colab ls
```

### 4. Launch the detached job

The launcher installs deps then spawns training in the background:

```bash
colab exec -f launch.py --timeout 120
```

The `--timeout 120` gives pip time to install torch, torchvision, and wandb (they are large). The launcher prints `OK. PID=<pid> log=/content/train.log` and returns -- training continues running because `start_new_session=True` detaches it from the Colab kernel.

### 5. Monitor progress

After giving training a minute to warm up:

```bash
colab exec -f check_progress.py --timeout 15
```

This shows: whether the process is alive, the last 15 log lines, and any checkpoint files in `/content/checkpoints/`.

For ongoing monitoring from your laptop:

```bash
while true; do
  clear
  date
  colab exec -f check_progress.py --timeout 15
  sleep 300
done
```

### 6. Download results

Your script saves checkpoints every 10 epochs to `./checkpoints/` which resolves to `/content/checkpoints/` on the VM. Download them before the session ends:

```bash
colab download checkpoints/
# or individually:
colab download checkpoints/epoch_10.pt
colab download checkpoints/epoch_20.pt
colab download checkpoints/epoch_30.pt
```

To get the browser URL (e.g. to open the notebook UI):

```bash
colab url
```

### 7. Stop the session (after done)

```bash
colab stop
```

---

## Important: session lifetime

Free-tier Colab sessions auto-terminate after approximately 2-4 hours of total runtime. Your 3-hour training run should complete, but there is no warning before termination. Mitigations:

- **Download checkpoints periodically** during training, not just at the end.
- If training is cut off, create a new session, upload the latest checkpoint, and resume from it.
- The `colab download` command works while the session is still alive. After the session dies, all files are gone.

## If the session dies mid-training

```bash
# Confirm the session is gone
colab sessions

# Create a fresh session
colab new --gpu T4 --session cifar-run-2

# Re-upload everything, including the checkpoint to resume from
colab upload train_cifar.py train_cifar.py
colab upload launch.py launch.py
colab upload check_progress.py check_progress.py
colab upload checkpoints/epoch_20.pt checkpoints/epoch_20.pt

# Re-launch (your script should support --resume or load the checkpoint)
colab exec -f launch.py --timeout 120
```

## Gotchas to watch for

- **`colab exec -f` requires relative paths.** Upload to `/content/train_cifar.py` but exec with `-f train_cifar.py`.
- **SSL errors are transient.** If `colab exec` fails with an SSL error, run `colab sessions` first -- the session and background process are often still alive. Re-run the command, don't create a new session.
- **Stale local cache.** `colab` caches session info locally. Run `colab sessions` to refresh from the server if commands mysteriously fail.
- **stdout buffering.** The `launch.py` template already handles this (`-u` + `PYTHONUNBUFFERED=1`). If you write a custom launcher, do not skip this.
