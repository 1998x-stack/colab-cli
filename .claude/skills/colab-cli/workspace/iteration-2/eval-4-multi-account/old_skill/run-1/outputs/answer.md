# Running parallel GPU sessions

The error `TooManyAssignmentsError: 412 Precondition Failed` occurs because Colab's free tier **limits you to one GPU session per Google account**. When you already have `vision-training` running on your primary account (`hackxie1998@gmail.com` via the `colab` alias), provisioning a second GPU session on the same account is rejected.

## Solution: Use multi-account aliases

This machine has three Google accounts configured with isolated `$HOME` directories. Each acts as a completely independent Colab user, giving you a separate GPU slot per account:

| Alias | Account | Effective HOME |
|-------|---------|---------------|
| `colab` | hackxie1998@gmail.com | default `~` |
| `cb`   | stefaniehu929@gmail.com | `~/colab-accounts/account-b` |
| `cc`   | xbetterdetermine@gmail.com | `~/colab-accounts/account-c` |

Since `vision-training` is already running on `colab` (account A), provision the NLP session on account B:

```bash
cb new --gpu T4 -s nlp-training
cb upload train_nlp.py /content/train_nlp.py
cb exec -f scripts/launch_proxy.py --timeout 120
```

Both sessions now run in parallel on separate Colab VMs, each billed to a different Google account.

## Managing both sessions

```bash
# Check status of each independently
colab status -s vision-training
cb status -s nlp-training

# List all sessions
colab sessions
cb sessions

# Stop them independently
colab stop -s vision-training
cb stop -s nlp-training
```

## Verify accounts

```bash
colab whoami   # hackxie1998@gmail.com
cb whoami      # stefaniehu929@gmail.com
```

## Alternative approach: Sequential execution

If you don't have a second account, you must run the jobs sequentially:

1. Train vision, download checkpoints, stop the session.
2. Create a new session, train NLP, download checkpoints, stop.

This avoids the one-GPU-per-account limit but obviously does not give you parallelism.

## Why this happens

Colab free tier allocates at most one GPU runtime per user. The `412 Precondition Failed` with `TooManyAssignmentsError` is the server-side rejection of a second concurrent GPU assignment. This is a Colab policy constraint, not a bug in the CLI.
