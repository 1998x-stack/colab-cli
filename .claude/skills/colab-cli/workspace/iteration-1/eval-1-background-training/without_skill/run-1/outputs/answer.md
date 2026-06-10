# Running a Background Training Job on Colab GPU with colab-cli

This guide covers running `~/ml/train_cifar.py` (3-hour training, needs `torch`, `torchvision`, `wandb`, saves checkpoints every 10 epochs to `./checkpoints/`) on a Colab GPU so it persists even if your laptop goes to sleep.

---

## How It Works

The colab-cli provisions a VM on Google's infrastructure -- not your laptop. The training runs remotely. The only local process is a lightweight keep-alive daemon that sends a ping every 60 seconds to prevent the session from being reclaimed. If your laptop sleeps, the training continues on Google's side; when you wake up, the daemon resumes pinging.

---

## Prerequisites

### 1. Install colab-cli

```bash
pip install google-colab-cli
```

### 2. Authenticate (ADC mode, most reliable for agents/headless)

The Colab runtime service requires specific OAuth scopes. Run this once:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Verify your token is valid:

```bash
colab whoami          # prints email, scopes, expiry
colab sessions        # lists any existing server-side assignments
```

If you hit a 403 during provisioning, you are missing the `colaboratory` scope -- re-run the `gcloud` command above.

---

## Step-by-Step Setup

### 1. Create a Session with GPU

```bash
colab new -s train-cifar --gpu T4
```

Session name is `train-cifar`. The daemon auto-starts in the background. The GPU options are `T4`, `L4`, `G4`, `H100`, `A100`; availability varies by subscription tier. Fall back to CPU (omit `--gpu`) if a GPU is unavailable.

### 2. Upload the Training Script

The default working directory on the VM is `/content`.

```bash
colab upload -s train-cifar ~/ml/train_cifar.py /content/train_cifar.py
```

### 3. Install Dependencies

```bash
colab install -s train-cifar torch torchvision wandb
```

This runs `uv pip install --system` on the VM, falling back to `pip`. The Colab VM has CUDA pre-installed, so `torch` will pick up the GPU automatically on first use.

### 4. Create a Launch Script (Detached Execution)

The key to background execution is `subprocess.Popen` with `start_new_session=True`. This detaches the child process from the Jupyter kernel so it survives the `colab exec` call finishing.

Create this as `launch_training.py` locally:

```python
"""Launch train_cifar.py in background on the Colab VM."""
import subprocess, sys, os

# Redirect output to a log file so we can tail it later
logfile = "/content/training.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # WANDB_API_KEY: set if wandb requires it
    # env["WANDB_API_KEY"] = "your-key-here"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train_cifar.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

# Print PID so we can pgrep later
print(f"Training launched. PID={proc.pid}")
print(f"Log: {logfile}")
```

### 5. Launch the Training

```bash
colab exec -s train-cifar -f launch_training.py
```

This returns immediately (the launch script finishes once Popen returns). The training runs in the background on the remote VM. You can now close your laptop -- the training continues.

#### One-liner (no separate launch file)

If you prefer not to create `launch_training.py`, pipe the code directly:

```bash
colab exec -s train-cifar << 'PYEOF'
import subprocess, sys, os
logfile = "/content/training.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train_cifar.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"Training launched. PID={proc.pid}")
print(f"Log: {logfile}")
PYEOF
```

---

## Monitoring Progress

### Check if the process is running

```bash
colab exec -s train-cifar << 'PYEOF'
import subprocess
result = subprocess.run(["pgrep", "-f", "train_cifar"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Training RUNNING -- PID(s): {result.stdout.strip()}")
else:
    print("Training NOT RUNNING")
PYEOF
```

### Tail the log

```bash
colab exec -s train-cifar << 'PYEOF'
with open("/content/training.log") as f:
    lines = f.readlines()
    for line in lines[-20:]:
        print(line.rstrip())
PYEOF
```

### List checkpoints

```bash
colab exec -s train-cifar << 'PYEOF'
import os
ckpt_dir = "/content/checkpoints"
if os.path.exists(ckpt_dir):
    for f in sorted(os.listdir(ckpt_dir)):
        size_kb = os.path.getsize(os.path.join(ckpt_dir, f)) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
else:
    print("(no checkpoints yet)")
PYEOF
```

---

## Downloading Checkpoints

```bash
# Download individual checkpoints
colab download -s train-cifar /content/checkpoints/epoch_10.pt ./checkpoints/epoch_10.pt
colab download -s train-cifar /content/checkpoints/epoch_20.pt ./checkpoints/epoch_20.pt

# Or download the log for analysis
colab download -s train-cifar /content/training.log ./training.log
```

The `colab download` command is one file at a time. For bulk downloads, use a script:

```bash
colab exec -s train-cifar << 'PYEOF'
import os, json, base64
ckpt_dir = "/content/checkpoints"
if os.path.exists(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f))]
    manifest = {}
    for f in files:
        path = os.path.join(ckpt_dir, f)
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
            manifest[f] = b64
    print(json.dumps(manifest))
PYEOF
```

Then decode each file locally from the printed JSON.

---

## Cleanup (Critical -- Avoid Burning Compute Units)

Idle VMs consume Colab compute units. Always stop when done:

```bash
colab stop -s train-cifar
```

This kills the keep-alive daemon, shuts down the kernel, unassigns the VM, and removes local session state.

---

## Important Caveats

- **Keep-alive cap:** The daemon runs for a maximum of 24 hours. Your 3-hour training fits easily within this window.
- **Laptop sleep behavior:** The keep-alive daemon is a local process. If your laptop sleeps for an extended period (hours), the daemon stops sending pings. The VM session may be reclaimed after a period of inactivity. For a 3-hour job, this is unlikely to be a problem -- the training starts before you close the laptop and finishes well before any reclamation.
- **Session state persists across `colab exec` calls:** Each call reattaches to the same kernel. Imports, variables, and background processes survive between calls. The kernel does not reset unless you explicitly `colab restart-kernel` or `colab stop`.
- **`colab run` is NOT suitable here:** `colab run` provisions, executes, and tears down in one synchronous call. It blocks until the script finishes. If your laptop sleeps mid-execution, the CLI connection drops and the command is interrupted. Use `colab new` + `colab exec` (with detached subprocess) instead.
- **wandb login:** If wandb requires interactive login, set `WANDB_API_KEY` as an environment variable in the launch script (commented out in the example above).
- **Disk quota:** `/content` on Colab VMs typically has ~166 GB available. Ensure your checkpoints fit.
