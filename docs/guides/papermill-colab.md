# Papermill on Colab

Execute parameterized Jupyter notebooks on Colab GPU/TPU VMs via papermill. Covers single runs and parameter sweeps.

## Why papermill + Colab

Papermill is the industry-standard tool for scripted notebook execution. It injects parameters into notebooks via tagged cells, runs them headlessly, and saves the executed output. On Colab, this means:

- **No notebook UI needed** — execute from terminal, cron, or CI
- **Parameter injection** — same notebook, different hyperparameters, without editing code
- **Reproducible outputs** — each run produces an executed `.ipynb` with embedded results
- **Works with Colab's detached subprocess pattern** — papermill runs in a process group that survives WebSocket drops

Compared to `jupyter nbconvert --execute`: papermill has parameter injection, better error messages, and per-cell timeout control. It's the right tool when you need to vary inputs across runs.

## How papermill works

1. You tag a cell in your notebook with `parameters` (in Jupyter: View → Cell Toolbar → Tags)
2. Papermill injects values into that cell before execution
3. It runs all cells top-to-bottom in a fresh kernel
4. The executed notebook is saved to the output path (with injected parameters visible)

### Parameter cell example

```python
# This cell is tagged "parameters"
epochs = 10
batch_size = 32
learning_rate = 0.001
dataset = "cifar10"
```

Then from the command line:
```bash
papermill input.ipynb output.ipynb -p epochs 50 -p batch_size 64 -p learning_rate 0.0005
```

Papermill overwrites the tagged cell with the injected values, then executes the full notebook.

## Colab workflow

The standard three-step pattern:

```bash
# 1. Provision a GPU VM
colab new --gpu T4 -s nb-run

# 2. Upload notebook + launch script
colab upload notebook.ipynb /content/notebook.ipynb
colab exec -f scripts/run_notebook.py --timeout 30

# 3. Download results
colab download /content/output.ipynb ./output.ipynb
colab stop -s nb-run
```

### The launcher script (`scripts/run_notebook.py`)

A minimal launcher that pip-installs papermill, spawns it as a detached subprocess, and returns immediately. This is the same detached-bootstrap pattern as Colab training — papermill runs in a process group that survives the `colab exec` WebSocket drop.

Edit the template before launching:
```python
NOTEBOOK = "notebook.ipynb"       # Input notebook (on VM at /content/)
OUTPUT = "output.ipynb"           # Output notebook path
PARAMS = {"epochs": 50, "lr": 0.001}  # Parameters to inject
DEPS = ["numpy", "matplotlib"]    # Extra pip packages
```

### Monitoring execution

Since papermill runs detached, use `check_progress.py` or a cron watchtower to monitor:

```bash
# Quick check — is papermill still running?
echo 'import subprocess; r=subprocess.run(["pgrep","-f","papermill"],capture_output=True,text=True); print("Running" if r.stdout else "Done")' | colab exec -s nb-run

# Tail the log
echo 'import subprocess; subprocess.run(["tail","-5","/content/papermill.log"])' | colab exec -s nb-run
```

Papermill writes progress to stderr (cell number, execution time). Redirect to a log file for monitoring.

## Parameter sweeps

For hyperparameter search, run the same notebook with different parameters sequentially:

```bash
colab upload notebook.ipynb /content/notebook.ipynb
colab exec -f scripts/param_sweep.py --timeout 30
```

The `scripts/param_sweep.py` template accepts a parameter grid and runs papermill once per combination, saving each output with a distinct filename (e.g., `output_epochs50_lr0.001.ipynb`).

## Gotchas

- **Papermill needs `ipykernel` installed.** Colab VMs have it by default, but bare Python environments may not. `pip install papermill ipykernel` is the safe bet.
- **The parameters cell must be a code cell, not markdown.** Papermill looks for a code cell with the `parameters` tag.
- **Papermill overwrites the parameters cell completely.** Any code in that cell besides the default variable assignments is lost after injection. Keep the parameters cell clean — only defaults.
- **Parameter values are injected as literals.** Strings get quoted, numbers don't. `-p name "hello"` → `name = "hello"` in the cell. `-p epochs 50` → `epochs = 50`.
- **Large notebooks (>100MB output) may timeout on download.** Colab's REST download has a practical limit around 600MB through the proxy. For notebooks with heavy embedded output (plots, images), consider saving figures to files instead of inline display.
- **Papermill creates a new kernel per run.** No state leaks between runs — each is a clean Restart & Run All.
- **Colab free-tier GPU sessions die after ~10 min.** Ensure your notebook completes within that window. Use `--ExecutePreprocessor.timeout` or papermill's own timeout to avoid hanging cells eating the session window.
