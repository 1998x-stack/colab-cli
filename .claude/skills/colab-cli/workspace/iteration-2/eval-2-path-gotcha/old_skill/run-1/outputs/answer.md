# Answer: `colab exec -f` with absolute path gives FileNotFoundError

## The problem

`colab exec -f` reads a **local** file from the machine you're running the command on. It does **not** read a file from the remote Colab VM.

You ran:

```bash
colab exec -f /content/script.py
```

This tells the CLI to look for a file at `/content/script.py` on your **local machine** (your laptop/desktop). Since that path doesn't exist locally, you get `FileNotFoundError`.

The upload you did earlier (`colab upload script.py /content/script.py`) placed the file on the **remote VM** at `/content/script.py`, but `colab exec -f` doesn't reference the remote filesystem -- it reads a local file and sends its contents to the VM for execution.

## The fix

Run `colab exec -f` with a **relative path** pointing to the script on your local machine:

```bash
colab exec -f script.py --timeout 120
```

This reads `./script.py` from your current working directory and sends it to the Colab VM for execution.

## The deeper picture

- `colab upload <local> <remote>` uploads a file **to** the VM (stored at the remote path).
- `colab exec -f <path>` reads a file **locally** and pipes it over a WebSocket for execution.
- The only reason to upload a script (`colab upload`) before executing it is if your exec'd script will spawn a **subprocess** that needs that uploaded file. For example, you upload `train.py` to `/content/train.py`, then `colab exec -f launch.py` runs `launch.py` (locally read), which in turn spawns `python /content/train.py` as a detached subprocess on the VM.

This is documented in the colab-cli SKILL.md:

> **`colab exec -f` reads LOCAL files (relative to CWD), not remote VM files.** Upload is only needed for scripts spawned as subprocesses by the exec'd script. `cd` to the right directory before `colab exec -f`.
