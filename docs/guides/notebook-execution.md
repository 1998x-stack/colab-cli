---

### Method 1: Use `jupyter nbconvert` (built-in, no extra install)
This is Jupyter's official native tool. It launches a fresh kernel and runs every cell sequentially from top to bottom — equivalent to **Restart & Run All** in the notebook UI.

#### Basic end-to-end execution
First use `cd` in cmd to navigate to your notebook's folder, then run:
```cmd
jupyter nbconvert --to notebook --execute your_file.ipynb
```
- `--execute`: the core flag that triggers full cell execution
- `--to notebook`: saves output as a `.ipynb` file with all results, prints and plots embedded. Without this, it defaults to exporting an HTML file.

By default it generates a new output file named `your_file.nbconvert.ipynb`.

#### Common optional flags
- `--inplace`: overwrite the original notebook file with executed results
  ```cmd
  jupyter nbconvert --to notebook --execute --inplace your_file.ipynb
  ```
- `--allow-errors`: continue running remaining cells even if one cell throws an error
- `--ExecutePreprocessor.timeout=600`: set per-cell timeout in seconds (default is 30s; increase for long-running tasks)
- `--ExecutePreprocessor.kernel_name=python3`: specify which Jupyter kernel / conda environment to use

---

### Method 2: Use `papermill` (best for automation / parameterized runs)
`papermill` is the industry-standard tool for scripted and automated notebook execution. It supports parameter injection and works very well in pipelines.

1. Install it first:
   ```cmd
   pip install papermill
   ```

2. Basic end-to-end run:
   ```cmd
   papermill input_notebook.ipynb output_notebook.ipynb
   ```
It runs all cells in order and saves the fully executed notebook to the output path.

3. Pass parameters into the notebook
Tag a cell in your notebook with the `parameters` label, then inject values directly from the command line:
```cmd
papermill input.ipynb output.ipynb -p epochs 50 -p dataset "data.csv"
```

---

### Method 3: Create a Windows batch (.bat) script
For one-click or scheduled execution, save your commands into a `.bat` script file.

#### Single notebook script (`run_notebook.bat`)
```bat
@echo off
:: Navigate to your notebook folder
cd /d "C:\path\to\your\notebook_folder"

:: Optional: activate your virtual / conda environment first
:: call activate your_env_name

:: Execute notebook end-to-end and overwrite original file
jupyter nbconvert --to notebook --execute --inplace my_analysis.ipynb

echo.
echo Notebook execution completed.
pause
```
You can double-click this `.bat` file, or call it from other scripts.

#### Batch-run all notebooks in a folder
```bat
@echo off
cd /d "C:\path\to\notebooks_folder"
for %%f in (*.ipynb) do (
    echo Running: %%f
    jupyter nbconvert --to notebook --execute --inplace "%%f"
)
echo All notebooks finished.
pause
```

---

### Important Notes
1. **Environment check**: Ensure `jupyter` is a recognized command in cmd. If not, activate your Python/conda environment first, or use the full path to your Python `Scripts` directory.
2. **Working directory**: Relative file paths inside your notebook will resolve relative to cmd's current working directory. Always `cd` to the notebook's folder before running.
3. **Clean state**: Both `nbconvert` and `papermill` start a brand-new kernel for every run — there is no leftover variable state, so you get a true clean end-to-end execution.

