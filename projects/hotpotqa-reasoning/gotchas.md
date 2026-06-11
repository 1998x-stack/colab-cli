# Gotchas — HotpotQA Reasoning Comparison

Field-tested failure modes from 7 Colab deployment attempts. Read before re-running.

## vLLM CUDA mismatch on Colab T4

**Symptom:** `ImportError: libcudart.so.13: cannot open shared object file: No such file or directory`

**Why:** Colab T4 runs CUDA 12.8 with PyTorch 2.11.0+cu128. vLLM >=0.8.0 wheels on PyPI are CUDA 13 only. Even with `--extra-index-url https://download.pytorch.org/whl/cu128`, pip pulls the default CUDA 13 wheel from PyPI because vLLM doesn't publish on the PyTorch index.

**Fix:** Don't use vLLM on Colab T4. Use HuggingFace `transformers` + `AutoModelForCausalLM` directly. The AWQ model (~4.5GB) fits comfortably in 16GB VRAM. No continuous batching, but for ≤50 examples it's fast enough.

If you must use vLLM, install a pre-0.8.0 version (`vllm==0.7.3`) which has CUDA 12.x wheels. But this creates `transformers` version conflicts with Colab's pre-installed packages.

## datasets >=4.0 breaks hotpot_qa

**Symptom:** `HfUriError: Repository id must be 'namespace/name', got 'hotpot_qa'`

**Why:** Colab ships `datasets==4.0.0` which changed how dataset paths are parsed. The `hotpot_qa` dataset (and many older HF datasets) use a format incompatible with datasets 4.x.

**Fix:** Pre-download the data locally and upload the JSON file. Then no `datasets` library is needed on the VM.

```bash
python -c "
from datasets import load_dataset
import json, random
ds = load_dataset('hotpot_qa', 'distractor', split='validation')
rng = random.Random(42)
indices = rng.sample(range(len(ds)), 50)
# ... extract and save
"
```

If you must use datasets on Colab, install `datasets==2.21.0` + `huggingface-hub==0.23.0`. But this creates version conflicts with anything that needs a newer `huggingface-hub` (transformers, vLLM, etc.).

## AWQ models need gptqmodel on recent transformers

**Symptom:** `ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with 'pip install gptqmodel'`

**Why:** Newer `transformers` versions moved AWQ support from `autoawq` to `gptqmodel`. `autoawq` alone is not sufficient.

**Fix:** Install both: `pip install autoawq gptqmodel`.

## Free-tier T4 sessions die in ~15 minutes

**Symptom:** Session disappears mid-run. `colab exec` returns 404/401. `colab ls` shows "Session not found."

**Why:** Google terminates free-tier GPU sessions after ~12-15 minutes regardless of activity. No warning, no recovery. All files on the VM are lost.

**Fix:** Keep the full pipeline under 10 minutes total. For this project:
- pip install: 60-90s
- Model download (first time): 2-5 min for 4.5GB AWQ model
- Experiment: depends on example count
  - With vLLM (batched): 50 examples ≈ 2-4 min
  - With transformers (sequential): each example ≈ 3-5s for CoT, 5-15s for ReAct
- **10 examples with transformers fits comfortably.** 50+ needs vLLM or multi-session.

## colab exec WebSocket drops during long-running commands

**Symptom:** `RuntimeError: Connection was lost` during pip install or model download.

**Why:** The Jupyter kernel WebSocket connection is flaky for operations >30s. The SOCKS5 proxy (ALL_PROXY) can interfere with WebSocket connections even when `no_proxy` is set for `*.colab.dev`.

**Fix:** Always spawn long-running work as detached subprocesses via `start_new_session=True`. The exec just spawns and returns immediately. Monitor progress via `colab ls` (REST API, no WebSocket) or `colab download` to read log files.

```python
# In your launch script:
p = subprocess.Popen(
    [sys.executable, "-u", "/content/run_all.py"],
    stdout=logfile, stderr=subprocess.STDOUT,
    start_new_session=True,
)
```

## Proxy flakiness from China

**Symptom:** Intermittent `SSLError`, `ReadTimeout`, or `Connection was lost` errors.

**Why:** Google Colab APIs are blocked in mainland China. Routing through Clash/Meta proxy (mixed-port 7890) is required, but the proxy itself can be unreliable. WebSocket connections are particularly sensitive.

**Fix:**
- REST API calls (`colab new`, `colab ls`, `colab upload`, `colab download`): use `HTTPS_PROXY=http://127.0.0.1:7890`
- WebSocket calls (`colab exec`): add `no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"` to bypass proxy for kernel connections
- If exec still drops, try removing `ALL_PROXY` entirely and using only `HTTPS_PROXY`/`HTTP_PROXY`
- Transient errors are normal — retry before assuming session is dead

## Multi-account rate limits

**Symptom:** `TooManyAssignmentsError: Precondition Failed` when provisioning a new GPU session.

**Why:** Free tier allows only 1 GPU per account. Rapid provisioning across sessions can trigger rate limits even across different accounts.

**Fix:** Wait 30-60 min between provisioning cycles. The 3 accounts (colab, cb, cc) let you run 3 sessions in parallel, but only if provisioned with gaps between them.

## colab upload can't create subdirectories

**Symptom:** HTTP 500 when uploading to `/content/strategies/cot.py` if the `strategies/` directory doesn't exist on the VM.

**Fix:** Upload flat to `/content/` root, then create directories via `colab exec` and move files. Or use a monolithic script that writes all files inline.

## colab exec stdin: avoid f-strings and special characters

**Symptom:** `SyntaxError: invalid syntax` or `unterminated string literal` when piping Python code to stdin.

**Why:** The shell interprets `$`, `\`, `{}`, and quotes before Python sees them. Complex inline scripts break unpredictably.

**Fix:** For anything beyond simple expressions (`print("hello")`), write a script file and use `colab exec -f script.py` or upload it and spawn detached.
