# Colab Session SSL Error Recovery (PID 903)

## Is the training dead?

**Not necessarily.** SSL errors from `colab exec` are often a proxy/network issue, not a sign that the session or your background process (PID 903) has died.

## Root cause (most likely)

Per the colab-cli skill (proxy setup section), the `SSLError: UNEXPECTED_EOF_WHILE_READING` error is a known symptom of proxy misconfiguration. Google Colab APIs are blocked from mainland China, so `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` must be set. However, the WebSocket kernel connection (`*.colab.dev`) used by `colab exec` often breaks over SOCKS5 — producing exactly this error.

The fix depends on which proxy variant your session needs:

| Symptom | Fix |
|---------|-----|
| `SSLError: UNEXPECTED_EOF_WHILE_READING` on `colab exec` | Keep `HTTPS_PROXY`/`HTTP_PROXY`, but exclude WebSocket domains from the proxy: `export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"` |
| All `colab` commands fail | Proxy is down or not set. Re-export: `HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890` |
| `RuntimeError: Connection was lost` | Same as the SSL error fix above — add the `no_proxy` exclusion for `*.colab.dev` |

The "which variant works" changes per session — if one combination fails, try the other.

## Recovery procedure (do these in order)

### Step 1: Check if the session is still alive

SSL errors affect `colab exec` but usually not the management commands:

```bash
colab sessions
colab status -s training
```

If `colab sessions` shows your session and `colab status` shows IDLE or BUSY, the VM is still running. Your training process (PID 903) is likely still alive on the VM even though `colab exec` cannot connect a WebSocket.

### Step 2: Fix the proxy config and reconnect

```bash
# Set up proxy
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

# Exclude WebSocket domains from proxy (this is likely what you need)
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"

# Test connection
colab exec -f check_progress.py --timeout 15
```

If that still fails, try without `no_proxy` — the skill notes that different sessions need different proxy variants.

### Step 3: Verify training is still running

Once `colab exec` works again, run your check script or:

```bash
echo 'import subprocess; subprocess.run(["pgrep", "-f", "sac_mountaincar"])' | colab exec --timeout 10
```

Or check the log:

```bash
echo 'with open("/content/sac_train.log") as f: print(f.read())' | colab exec --timeout 10
```

### Step 4: Download checkpoints immediately

Session lifetimes on free tier are ~2-4 hours. Your session might be nearing its limit. Pull checkpoints now:

```bash
colab exec -f - --timeout 30 <<'PYEOF'
import subprocess, tarfile, os
# Check if checkpoints exist
ckpt_dir = "/content/checkpoints"
if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
    print(f"Checkpoints found: {os.listdir(ckpt_dir)}")
else:
    print("No checkpoints found")
PYEOF
```

Then download:

```bash
# Tar and download
echo 'import tarfile; tarfile.open("/content/ckpts.tar.gz", "w:gz").add("/content/checkpoints")' | colab exec --timeout 30
colab download /content/ckpts.tar.gz .
```

## What if the session is dead?

If `colab sessions` shows nothing for your `training` session:

1. **The VM is gone** — free tier GPU sessions auto-terminate after ~2-4 hours. Your background process and checkpoints are lost unless you already downloaded them.
2. **Recovery is impossible from the same session.** Start fresh:
   - If you have checkpoints on disk (from a prior `colab download`), create a new session and resume training from the latest checkpoint. SAC's resume logic in `sac_mountaincar.py` automatically loads the latest checkpoint from `/content/checkpoints/`.
   - If you never downloaded them, the training data is lost with the VM.

## Key takeaways from the skill

- **SSL errors are often transient** (gotcha #9). Don't assume death on first error.
- **colab exec uses WebSockets** which break over SOCKS5 — the `no_proxy` exclusion for `*.colab.dev` is the most likely fix.
- **Free tier sessions are ephemeral** — download checkpoints frequently. Each session lasts ~2-4 hours and can die in <30 minutes during peak times.
- **colab sessions and colab stop use REST API, not WebSockets**, so they often work when `colab exec` does not.
