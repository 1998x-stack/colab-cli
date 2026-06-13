---
name: papermill-colab
description: >
  Use when executing Jupyter notebooks on Colab VMs via papermill — running
  parameterized notebooks with GPU/TPU, hyperparameter sweeps, or batch
  notebook execution from the terminal. Triggers on mentions of papermill,
  notebook execution on Colab, running .ipynb files on GPU, parameterized
  notebook runs, or converting notebook-based workflows to automated Colab
  pipelines. Also trigger when the user wants to run the same notebook with
  different parameters, do hyperparameter search via notebooks, or automate
  notebook execution without the Colab UI.
---

# Papermill on Colab

Execute parameterized Jupyter notebooks on Colab GPU/TPU VMs via [papermill](https://github.com/nteract/papermill). The colab-cli skill handles VM provisioning, file transfer, and session management — this skill adds the notebook execution layer on top.

## Mental model

Papermill runs Jupyter notebooks headlessly with parameter injection. You tag a cell with `parameters` in the notebook, and papermill overwrites that cell with CLI-provided values before execution. It's a clean kernel start → inject params → run all cells → save output.

On Colab, papermill runs inside a **detached subprocess** on the VM — same pattern as training scripts. The launcher (`scripts/run_notebook.py`) pip-installs papermill, spawns it with `start_new_session=True`, and returns immediately. Papermill keeps running even after the `colab exec` WebSocket drops.

**When to use papermill vs. `colab exec -f script.py`:**
- Notebook has inline visualizations you want to preserve in the output → papermill
- Same notebook needs to run with different hyperparameters → papermill parameter injection
- You already have a working notebook and don't want to extract code to a .py file → papermill
- Pure training script with no notebook-specific features → `colab exec -f` (simpler, fewer deps)

## Quick reference

```bash
# Single notebook run on Colab GPU
colab new --gpu T4 -s nb-run
colab upload notebook.ipynb /content/notebook.ipynb
colab exec -f scripts/run_notebook.py --timeout 30   # spawns papermill detached
# ... wait for completion ...
colab download /content/output.ipynb ./output.ipynb
colab stop -s nb-run

# Parameter sweep
colab upload notebook.ipynb /content/notebook.ipynb
colab exec -f scripts/param_sweep.py --timeout 30
```

## The parameters cell

Papermill requires a code cell tagged `parameters` in your notebook. In Jupyter UI: View → Cell Toolbar → Tags, then type `parameters` in the tag field.

```python
# This cell must be tagged "parameters"
epochs = 10
batch_size = 32
learning_rate = 0.001
```

Papermill replaces the entire cell content with the injected values before execution. Keep this cell clean — only default values, no logic.

## Scripts

### `scripts/run_notebook.py` — single notebook execution

Edit the template before launching:

```python
NOTEBOOK = "notebook.ipynb"           # Input notebook path on VM (/content/)
OUTPUT = "output.ipynb"               # Output notebook path on VM
PARAMS = {"epochs": 50, "lr": 0.001}  # Parameters to inject (or {})
DEPS = ["numpy", "matplotlib"]        # Extra pip packages
LOG = "/content/papermill.log"        # Where papermill stderr goes
```

**Workflow:**
1. Upload notebook to VM: `colab upload notebook.ipynb /content/notebook.ipynb`
2. Edit `run_notebook.py` to set NOTEBOOK, OUTPUT, PARAMS, and DEPS
3. Launch: `colab exec -f scripts/run_notebook.py --timeout 30`
4. The launcher pip-installs papermill + deps, spawns papermill detached, and prints the PID
5. Monitor via `check_progress.py` or direct log tail

### `scripts/param_sweep.py` — hyperparameter sweep

Runs the same notebook sequentially with each parameter combination. Edit the template:

```python
NOTEBOOK = "notebook.ipynb"
PARAM_GRID = [
    {"epochs": 30, "lr": 0.01},
    {"epochs": 30, "lr": 0.001},
    {"epochs": 50, "lr": 0.01},
    {"epochs": 50, "lr": 0.001},
]
DEPS = ["numpy", "matplotlib"]
LOG = "/content/sweep.log"
```

Each run produces a distinct output file: `output_epochs30_lr0.01.ipynb`, etc.

## Monitoring

Papermill runs detached, so `colab exec` returns while execution continues. Monitor progress:

```bash
# Check if papermill is alive
echo 'import subprocess; r=subprocess.run(["pgrep","-f","papermill"],capture_output=True,text=True); print("Running" if r.stdout else "Done")' | colab exec -s nb-run

# Tail the log
echo 'import subprocess; subprocess.run(["tail","-10","/content/papermill.log"])' | colab exec -s nb-run

# Check output file size (growing = running)
colab ls -s nb-run | grep output
```

For long-running sweeps, set up a cron watchtower (see colab-cli skill, "Cron watchtower" section).

## Downloading results

Papermill output notebooks contain all cell outputs (text, plots, HTML) embedded. Download the executed notebook:

```bash
colab download /content/output.ipynb ./output.ipynb
```

For parameter sweeps, tar all outputs first:
```bash
echo 'import subprocess; subprocess.run(["tar","-czf","/content/sweep_results.tar.gz","-C","/content","."])' | colab exec -s nb-run
# Wait for tar to complete, then:
colab download -s nb-run /content/sweep_results.tar.gz ./results.tar.gz
```

## Gotchas

See `references/gotchas.md` for field-tested surprises. The critical ones:

1. **Papermill needs `ipykernel`** — Colab VMs have it, but if you ever get `No such kernel`, `pip install ipykernel`.
2. **The parameters cell must be a code cell** — markdown cells with the `parameters` tag are ignored. Papermill silently skips them and runs with defaults.
3. **Papermill overwrites the parameters cell entirely** — any code beyond default assignments is lost. Keep the cell to only variable defaults.
4. **All papermill output goes to stderr** — redirect stderr to capture progress. The launcher scripts handle this via `subprocess.Popen(stderr=subprocess.STDOUT)`.
5. **Notebooks with `!pip install` cells work but are fragile** — papermill runs those cells, but Colab's pre-installed packages may conflict. Prefer declaring deps in the launcher script's `DEPS` list.
