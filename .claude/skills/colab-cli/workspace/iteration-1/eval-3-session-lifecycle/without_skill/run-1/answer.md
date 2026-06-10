# Answer: SSL Errors on `colab exec`

## Your training is likely still alive. Do not panic.

SSL errors with `colab exec` are usually transient network issues. The session and your background process (PID 903) may be perfectly fine.

## Step-by-step recovery

### 1. Check session health first

Run this before anything else:

```bash
colab sessions && colab status
```

- If the session is listed and `colab status` returns valid info, the session is still alive. Your training process is probably still running. Retry `colab exec` -- it will likely work.
- If the session is listed but `colab status` shows an expired or stale URL, the session cache is stale. The `colab sessions` command refreshes from the server.

### 2. Verify the background process survived

If the session is alive, check whether PID 903 is still running:

```bash
echo 'import os, signal; print(os.kill(903, 0) is None)' | colab exec --timeout 10
```

Or check its log file (if you used the `launch.py` template or similar nohup pattern):

```bash
colab exec -f check_progress.py --timeout 15
```

### 3. If the session is dead

If `colab sessions` shows no active session, it was pruned (free Colab sessions last ~2-4 hours). The training process died with the VM. Your checkpoints at `/content/checkpoints/` are gone unless you ran `colab download` before.

**To recover whatever may be salvageable:**
- Check your local machine for any previously downloaded checkpoints.
- If you never downloaded them, the checkpoint directory was destroyed with the VM.

### 4. Resume training from the last saved checkpoint

If you have local checkpoints:

```bash
# Create a new session
colab new --gpu T4 --session my-training-v2

# Re-upload your code and your last checkpoint
colab upload train.py train.py
colab upload checkpoints/best.pt checkpoints/best.pt

# Re-launch with --resume flag or equivalent in your training script
colab exec -f launch.py --timeout 120
```

If you did not have checkpoint download automation in place, add it to your launch script for next time.

## Prevention for future runs

Add these safeguards before your next long training run:

1. **Download checkpoints periodically** during training by having your script upload via `colab upload` or save to a path you can pull.
2. **Use `colab run`** for shorter jobs -- it auto-teardowns but also auto-cleans.
3. **Monitor with a loop** so you catch session death quickly:

```bash
while true; do
  colab exec -f check_progress.py --timeout 15
  sleep 300
done
```

4. **Know the session timeout** -- free tier is ~2-4 hours total runtime, not wall-clock time. If your training needs longer, upgrade to Colab Pro/Pro+ or split into multiple sessions.

## Summary

| Scenario | Action |
|----------|--------|
| Session alive, SSL was transient | Retry `colab exec` -- training still running |
| Session dead, have local checkpoints | New session, re-upload code + checkpoint, resume |
| Session dead, no local checkpoints | Training lost. Re-run from scratch |
