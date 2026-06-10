# Session Lifecycle Recovery: SSL Errors on `colab exec`

## Your training is very likely not dead.

SSL errors (`SSLError: UNEXPECTED_EOF_WHILE_READING`) on `colab exec` are a documented transient issue. They happen intermittently and do **not** inherently mean the Colab session or your background process (PID 903) was destroyed. The session may still be alive and training may still be running.

However, **an hour is significant time on a free-tier session** (~2-4 hour lifetime). The session could also have been pruned silently after a connection error -- a known pattern where sessions die in under 5 minutes after exec failures.

## Step-by-step recovery

### 1. Verify session health

Run these to refresh the local cache and check the server-side state:

```bash
colab sessions && colab status
```

- **Session appears and `status` returns valid backend URL** -> The session is alive. SSL error was transient. Retry `colab exec` immediately to re-establish the WebSocket connection. Your training (PID 903) is likely still running if it was launched with `start_new_session=True`.
- **`colab sessions` shows the session or `status` fails** -> The session may have been pruned by the server.

### 2. Proxy-specific recovery (if applicable)

If you are behind a GFW proxy (mainland China), the WebSocket through SOCKS5 is known to be unstable:

```bash
# Try 1: Add no_proxy to bypass WebSocket domains through the proxy
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
colab exec -f check_progress.py --timeout 15

# Try 2: Retry without no_proxy (sometimes no_proxy breaks WebSocket routing)
unset no_proxy
colab exec -f check_progress.py --timeout 15
```

The correct combination varies per session -- flip and retry. `colab sessions`, `colab new`, and `colab stop` always need the proxy. Only `colab exec`, `colab download`, and `colab upload` might need `no_proxy` set.

### 3. If the session is dead: create new session and resume

```bash
# 1. Create a new GPU session
colab new --gpu T4 --session my-training-2

# 2. Re-upload your training artifacts
colab upload launch.py /content/launch.py
colab upload sac_mountaincar.py /content/sac_mountaincar.py
colab upload check_progress.py /content/check_progress.py

# 3. Restore checkpoints (if you downloaded them earlier)
colab download checkpoints/*.pt ./checkpoints/   # skipped if you already have them locally
colab upload checkpoints/best.pt /content/checkpoints/best.pt

# 4. Re-launch training
colab exec -f launch.py --timeout 120
```

The SAC training script (`sac_mountaincar.py`) automatically resumes from the latest checkpoint in `/content/checkpoints/` if one exists.

## What determines whether you lose progress?

| Scenario | Outcome |
|----------|---------|
| Session alive (step 1 confirms it) | Zero loss. PID 903 is still running. Retry `colab exec`. |
| Session dead, but you downloaded checkpoints previously | Resume from last saved checkpoint on new session. Some progress lost since last download. |
| Session dead, never downloaded checkpoints | Full loss of that run. Checkpoints at `/content/checkpoints/` are gone with the VM. |

## Why PID 903 matters

If your training was launched with `start_new_session=True` (required pattern in the SAC project's `launch.py`), PID 903 should survive kernel restarts and `colab exec` disconnections because it is in its own process group. If the session is alive, PID 903 is almost certainly still running. You can verify with `check_progress.py` which uses `pgrep` -- but note that `pgrep` can return false positives on Colab VMs matching kernel threads; always confirm with `ps -p 903` or look for the specific script name in the command line.

## Preventive measures for next time

- **Download checkpoints during the run**, not at the end. Use a monitoring loop:
  ```bash
  while true; do
    colab download /content/checkpoints/ ./checkpoints/
    sleep 300
  done
  ```
- Run `colab exec -f check_progress.py --timeout 15` periodically to catch session death early.
- If you hit SSL errors, always check `colab sessions` before assuming the worst -- transient SSL errors are the documented norm, not a sign of session death.
- For critical long-running workloads, consider Colab Pro for longer session lifetimes (up to 24 hours).
