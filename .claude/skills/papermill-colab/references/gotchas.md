# Papermill Gotchas

Field-tested surprises when using papermill, particularly on Colab VMs.

## Parameter injection

### The parameters cell must be a code cell

Papermill only recognizes `parameters` tags on code cells. A markdown cell tagged `parameters` is silently ignored — the notebook runs with default values and no error is raised.

**Symptom:** You pass `-p epochs 50` but the notebook uses the default `epochs = 10`.

**Fix:** In Jupyter, ensure the parameters cell type is "Code" (not "Markdown"). The tag toolbar shows cell type — the cell icon (</>) indicates code.

### Papermill overwrites the entire parameters cell

Papermill replaces the full cell content, not just the variable assignments. Any imports, comments, or helper code in the parameters cell are lost.

```python
# BEFORE injection (tagged "parameters"):
epochs = 10
batch_size = 32
# helper: import os; DATA_DIR = os.path.join(BASE, "data")  # LOST after injection

# AFTER `papermill ... -p epochs 50 -p batch_size 64`:
epochs = 50
batch_size = 64
```

Keep the parameters cell to **only** default variable assignments. Put imports and helper logic in a separate cell.

### Parameter values are Python literals

Papermill serializes parameters as Python literals:
- `-p epochs 50` → `epochs = 50` (int)
- `-p lr 0.001` → `lr = 0.001` (float)
- `-p name "cifar10"` → `name = "cifar10"` (str with quotes)
- `-p use_aug True` → `use_aug = True` (bool)

Strings get injected with surrounding quotes. If your parameter cell expects a bare string, papermill's injection is still correct — `name = "cifar10"` is valid Python.

### Raw strings and YAML parameters

Papermill supports `-r` for raw string parameters (no quoting) and `-y` for YAML-serialized parameters:

```bash
papermill input.ipynb output.ipynb -r text "hello world"  # → text = hello world (unquoted, likely SyntaxError)
papermill input.ipynb output.ipynb -y config '{"lr": 0.01, "epochs": 50}'  # → config = {'lr': 0.01, 'epochs': 50}
```

Prefer `-p` for simple scalars, `-y` for dicts/lists.

## Colab-specific

### ipykernel is required

Papermill uses `ipykernel` to execute notebooks. Colab VMs ship it by default, but if you see `No such kernel: python3`, install it:

```bash
pip install ipykernel
python -m ipykernel install --user --name python3
```

### GPU memory across papermill runs

Each papermill invocation starts a fresh kernel. GPU memory is released between runs — no leak across parameter sweep iterations. But within a single notebook run, cells accumulate GPU tensors as usual. Use `torch.cuda.empty_cache()` between cells if memory is tight.

### Notebook output size and download limits

Executed notebooks with inline plots and images can grow large (>100MB). The proxy download limit is ~600MB, but large .ipynb files may also hit session timeout during download. For notebook-heavy outputs, save figures to files instead of relying on inline display:

```python
# Instead of plt.show():
plt.savefig("/content/plots/epoch_{epoch}.png", dpi=72)  # lower dpi for smaller files
plt.close()
```

Then tar the plots directory for download.

### Colab's pre-installed packages can conflict

Colab VMs ship specific versions of numpy, matplotlib, pandas, etc. If your notebook `!pip install`-s a different version, subsequent cells may break. Pin versions in the launcher script's `DEPS` list rather than in notebook cells, so papermill runs in a clean environment.

## Workflow

### papermill output goes to stderr

All papermill progress output (cell execution status, timing, errors) goes to stderr, not stdout. The launcher scripts merge stderr into stdout for the log file:

```python
proc = subprocess.Popen(
    ["papermill", notebook, output, "-p", "epochs", "50"],
    stdout=f, stderr=subprocess.STDOUT,  # merge stderr → log
)
```

Without this, the log file contains only stdout from cells (print statements), not papermill's own progress info.

### Killing a stuck papermill run

If a cell hangs (e.g., infinite loop, deadlocked download), papermill won't exit. Kill it on the VM:

```bash
echo 'import subprocess, os, signal; \
procs = subprocess.run(["pgrep", "-f", "papermill"], capture_output=True, text=True); \
for pid in procs.stdout.strip().split(): \
    os.kill(int(pid), signal.SIGKILL)' | colab exec -s <session>
```

The partially-written output notebook may be corrupt — discard it.

### Parameter sweep disk usage

Each papermill run produces a full executed notebook. A 10-iteration sweep of a 5MB notebook produces ~50MB on disk. Ensure `/content/` has space (Colab gives ~68GB on free tier, but large sweeps with embedded images can fill it). Tar and download incrementally for large sweeps:

```python
import subprocess, glob

for nb in sorted(glob.glob("/content/output_*.ipynb")):
    subprocess.run(["tar", "-czf", nb.replace(".ipynb", ".tar.gz"), nb])
    # then download each .tar.gz individually
```
