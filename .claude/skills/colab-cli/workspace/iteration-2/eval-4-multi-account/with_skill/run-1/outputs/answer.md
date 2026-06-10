# Running Two Parallel GPU Training Jobs on Colab

## The Problem

You hit `TooManyAssignmentsError: 412 Precondition Failed` because **Google Colab's free tier permits only one GPU session per account at a time.** When you ran `colab new --gpu T4 -s vision-training`, that account's single GPU slot was claimed. A second `colab new --gpu T4` on the same account is rejected.

## The Solution: Multi-Account Aliases

The `colab` CLI doesn't natively support multiple accounts, but this machine has **three pre-configured aliases** in `~/.zshrc` that each use a fully isolated `$HOME` directory (separate OAuth2 tokens, sessions, and settings):

| Alias | Account | Purpose |
|-------|---------|---------|
| `colab` | hackxie1998@gmail.com | Your primary account (already hosting `vision-training`) |
| `cb` | stefaniehu929@gmail.com | Second account, isolated via `~/colab-accounts/account-b` |
| `cc` | xbetterdetermine@gmail.com | Third account, isolated via `~/colab-accounts/account-c` |

## Steps

Your `vision-training` session on the primary account is fine. To launch the second job in parallel:

```bash
# Keep vision-training running on the primary account.
# Use the second account for NLP training:
cb new --gpu T4 -s nlp-training
cb upload train.py /content/train.py
cb exec -f launch.py --timeout 120
# ... monitor, download checkpoints, etc.
cb stop -s nlp-training
```

You can verify each account's identity with `colab whoami` / `cb whoami` / `cc whoami`.

## Why It Works

Each alias overrides `$HOME` to an isolated directory (e.g., `~/colab-accounts/account-b`). All `colab` CLI state -- OAuth2 token at `~/.config/colab-cli/token.json`, session registry, settings, and logs -- lives under `$HOME`, so each alias authenticates and operates as a completely independent Google account. Google's one-GPU-per-account limit is satisfied because the two sessions belong to two different accounts.

## Cleanup

Both sessions auto-terminate after ~2-4 hours on the free tier. Always download checkpoints during the run, and stop idle sessions to conserve compute units:

```bash
colab stop -s vision-training
cb stop -s nlp-training
```

If you ever need a third parallel session, `cc` (third account) is available and works identically.
