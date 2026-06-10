# Path Gotcha: `colab exec -f` reads a LOCAL file, not a remote one

The `upload` and `exec -f` commands each interpret their path argument differently:

- **`colab upload script.py /content/script.py`** -- uploads a file from your **local** machine to the remote Colab runtime. The second argument (`/content/script.py`) is a **remote** path on the VM. This works correctly.

- **`colab exec -f /content/script.py`** -- `-f` means "read this file from the **local** filesystem and execute its contents on the remote runtime." The code calls `open(file, "r")` on your local machine (execution.py lines 129 and 141). Since `/content/script.py` does not exist on your local machine, you get `FileNotFoundError`.

## The fix

Run `exec` with the **local** path to your script, not the remote path:

```
colab exec -f script.py
```

The `exec` command reads `script.py` from your local disk and sends its contents to the Colab kernel for execution. The file does not need to be uploaded first -- `exec -f` handles the upload of the code content itself.

## Why the upload command exists separately

`upload` is for placing files on the remote filesystem (data files, other scripts that your code imports, etc.) where they persist across executions. `exec -f` is for running code -- it reads locally and streams the content to the kernel.
