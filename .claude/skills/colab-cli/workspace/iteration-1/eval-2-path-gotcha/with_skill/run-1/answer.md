`colab exec -f` requires a **relative path**, not an absolute one. The working directory on the Colab VM is `/content/`, so even though you uploaded to `/content/script.py`, you need to run it as:

```bash
colab exec -f script.py
```

Using `/content/script.py` causes a `FileNotFoundError` because the `-f` flag interprets the argument as a path relative to `/content/` -- it does not strip the `/content/` prefix, so it ends up looking for the file at an incorrect location.

In short: `colab upload` accepts absolute paths, but `colab exec -f` only works with paths relative to `/content/`.
