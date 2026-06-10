# Session Lifecycle Recovery: SSL Errors on `colab exec`

## Your training is very likely not dead.

SSL errors with `colab exec` are a known transient issue documented under gotcha #5. The error signature (`SSLError: UNEXPECTED_EOF_WHILE_READING`) happens intermittently and does **not** mean the session or your background process (PID 903) has been destroyed. The Colab backend URL can also change across kernel restarts, breaking the cached connection without killing the VM.

## Step-by-step recovery

### 1. Verify session is still alive

Run both commands — the first refreshes the local session cache from the server, the second shows current session status:

```bash
colab sessions && colab status
```

If the session appears in the list and `colab status` returns a valid backend URL, your session is alive and your training process (PID 903) is likely still running. The SSL error was a transient network hiccup. Try `colab exec` again.

### 2. If `colab sessions` returns empty or `colab status` fails

The session may have been pruned. Free-tier Colab sessions auto-terminate after ~2-4 hours of total runtime with no warning. An hour since it was working is within the window where this is possible, especially if the session was already old.

The local CLI may still show stale session info. `colab sessions` refreshes from the server — trust its output, not the cached state.

### 3. If the session is dead: create a new one and resume

```bash
# 1. Create a new session
colab new --gpu T4 --session my-training-2

# 2. Re-upload your training script
colab upload train.py train.py
colab upload launch.py launch.py
colab upload check_progress.py check_progress.py

# 3. Restore checkpoints if you downloaded them before the session died
colab upload checkpoints/best.pt checkpoints/best.pt

# 4. Re-launch training (your script should auto-detect existing checkpoints and resume)
colab exec -f launch.py --timeout 120
```

## What determines whether you lose progress?

- **If the session is still alive** (step 1 shows it): Zero loss. Re-run `colab exec` and PID 903 is still running.
- **If the session is dead but you downloaded checkpoints**: Re-upload them to the new session and resume from the last checkpoint.
- **If the session is dead and you never downloaded checkpoints**: Those checkpoints at `/content/checkpoints/` are gone. The VM is destroyed on session termination — files do not survive.

## Preventive measures for next time

- Download checkpoints **during** the run, not at the end. Use the monitoring loop pattern:
  ```bash
  while true; do
    colab download checkpoints/best.pt
    sleep 300
  done
  ```
- Use `scripts/check_progress.py` to verify PID 903 is alive and listing checkpoints.
- For critical runs, upgrade to Colab Pro for longer session lifetimes.
- Use `colab run` (ephemeral, auto-teardown) for CI-style batch jobs that don't need manual monitoring.
