# Colab Operations

File I/O, execution patterns, multi-account management, and monitoring.

## File I/O

### `colab exec -f` reads LOCAL files, not VM files

The `-f` flag reads Python from your LOCAL filesystem and sends it to the VM. It does NOT run files already on the VM. Upload is only needed for files spawned as subprocesses.

```bash
colab exec -f launch.py    # sends ./launch.py from local CWD
```

### Upload with relative path lands in /content

```bash
colab upload local.py remote.py     # → /content/remote.py
```

### `colab upload` creates a FILE not a directory

When `/content/myproject/` doesn't exist, uploading to `/content/myproject/script.py` creates a FILE named `/content/myproject`. All subsequent uploads overwrite it. **Fix:** Create dirs via exec first, or upload flat to `/content/` root.

### Multi-file deploy: base64 embed

For projects with many files, embed them in a single script to avoid multiple uploads:

```python
import os, base64
lines = ['import os, base64']
for fname in os.listdir(proj_dir):
    if fname.endswith('.py'):
        with open(fname) as f:
            encoded = base64.b64encode(f.read().encode()).decode()
        lines.append(f'with open("/content/{fname}", "w") as f:')
        lines.append(f'    f.write(base64.b64decode("{encoded}").decode())')
```

### Directory download not supported

Tar on VM first: `tar -czf /content/out.tar.gz -C /content dir/`, then `colab download`.

### Checkpoint downloads >600MB fail through proxy

Save separate weights-only checkpoint (~120-233MB) for download. Full checkpoint with optimizer state (~1GB) breaks at ~624MB with `IncompleteRead`.

## Execution

### Detached bootstrap for anything >30s

`colab exec` WebSocket drops during sustained operations. Spawn heavy work as detached subprocess:

```python
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open("/content/train.log", "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
```

Key: `PYTHONUNBUFFERED=1` + `python -u` (unbuffered output) + `start_new_session=True` (survives exec timeout).

### No `-c` flag

`colab exec` does not support `-c`. Use stdin pipe: `echo 'print("hello")' | colab exec --timeout 10`

### Avoid f-strings in stdin pipes

Shell interprets `$`, `\\`, `{}` before Python sees them. Use heredocs for multi-line code:

```bash
cat <<'PYEOF' | colab exec --timeout 30
# Python code here — safe from shell interpretation
PYEOF
```

### Cron watchtower for monitoring

REST-based download survives WebSocket drops. Pattern:
```bash
# Cron: every 2 min — tar on VM → download via REST → extract → report
echo 'subprocess.run(["tar", "-czf", "/content/out.tar.gz", "-C", "/content", "dir"])' | colab exec -s "$S" --timeout 15
colab download -s "$S" /content/out.tar.gz ./out.tar.gz
tar -xzf out.tar.gz && tail logs/train.log
```

## Multi-Account

Six accounts configured via `$HOME` isolation. Each alias overrides HOME to point at isolated directory tree.

| Alias | Account | HOME |
|-------|---------|------|
| `colab` | hackxie1998 | default `~` |
| `cb` | stefaniehu929 | `~/colab-accounts/account-b` |
| `cc` | xbetterdetermine | `~/colab-accounts/account-c` |
| `clb` | xieminghack | `~/colab-accounts/account-clb` |
| `clab` | xieminghacker | `~/colab-accounts/account-clab` |

Only 1 GPU session per account. GPU quota exhausts across ALL accounts simultaneously (12-24h cooldown). Spread GPU usage across days.

## Hardware

- Free tier: T4 usually available, TPU often rejected
- Pro/Pro+: L4, G4, sometimes H100/A100
- CPU (omit `--gpu`): always works
