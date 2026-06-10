# Why `colab exec -f /content/script.py` Gets FileNotFoundError

## The Problem

`colab exec -f` reads a Python file from your **local filesystem** and sends it to the VM for execution. It does **not** run files that already exist on the VM.

When you run:

```bash
colab exec -f /content/script.py
```

The CLI interprets `/content/script.py` as a **local path** on your machine. Since there is no `/content/` directory on your local computer, it fails with `FileNotFoundError`.

## The Confusion

This is a common gotcha because the flag name `-f` (which stands for "file") sounds like it should point to a file on the remote VM. In reality:

| Command | Reads from | Purpose |
|---------|-----------|---------|
| `colab upload <local> <remote>` | Local disk | Copies a local file to the VM |
| `colab exec -f <path>` | Local disk | Sends a local file to the kernel for execution |
| `colab download <remote> <local>` | Remote VM | Copies a VM file to local disk |

## The Fix

There are two cases depending on what you actually want to do:

### Case 1: Run the file already uploaded to `/content/script.py`

The `-f` flag cannot do this. Instead, pipe Python code to stdin:

```bash
echo 'exec(open("/content/script.py").read())' | colab exec
```

Or upload a small launcher script that imports or runs the target file.

### Case 2: Re-run the local script (re-send it)

Use the local path (relative to your CWD or absolute local path):

```bash
# From the directory containing script.py:
colab exec -f script.py

# Or with a local absolute path:
colab exec -f /Users/me/project/script.py
```

Note that `-f` sends the file to the kernel for execution each time -- no separate `colab upload` is needed for the file being exec'd. Upload is only necessary for auxiliary scripts that your main script spawns as subprocesses.

## Why Upload Exists

You might wonder: "If `-f` sends the file, why did my upload succeed?" The upload step is only required when your exec'd script spawns **subprocesses** that reference other files on the VM. For example:

```python
# launch.py (sent via colab exec -f launch.py -- no upload needed)
import subprocess
subprocess.Popen(["python", "/content/worker.py"], start_new_session=True)
```

Here, `worker.py` must be on the VM (via `colab upload`) before `launch.py` runs, because `-f` only sends `launch.py`, not its subprocess dependencies.

## Key Takeaway

**`colab exec -f` always reads from your local filesystem, never from the VM.** The path you pass to `-f` is relative to your local CWD (or an absolute path on your machine). If the file is already on the VM, you need a different approach (stdin pipe, or re-upload and exec from local).

Reference: `references/gotchas.md` section "Path handling" in the colab-cli skill.
