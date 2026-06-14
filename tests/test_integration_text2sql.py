"""End-to-end test: dataset -> train -> eval on 3 hand-crafted examples."""
import json
import os
import subprocess
import sys
import tempfile

PROJECT_DIR = "projects/nlp/text2sql_finetune"
# macOS OpenMP fix: avoid SIGABRT from duplicate libomp initialisation
_ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}


def test_e2e_minimal():
    """Run full pipeline with 3 examples. Verifies all modules connect correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_data.pt")
        lora_path = os.path.join(tmpdir, "lora_weights")
        eval_path = os.path.join(tmpdir, "eval_report.json")

        # 1. Create dataset (using real HF dataset, 3 examples)
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/dataset.py",
            "--split", "train", "--max_examples", "3",
            "--output", data_path,
        ], check=True, timeout=120, env=_ENV)

        assert os.path.exists(data_path), "dataset not saved"

        # 2. Train (3 examples, 2 steps)
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/train.py",
            "--data_path", data_path,
            "--output_dir", lora_path,
            "--max_steps", "2",
            "--batch_size", "2",
            "--grad_accum", "1",
        ], check=True, timeout=300, env=_ENV)

        assert os.path.exists(os.path.join(lora_path, "adapter_config.json")), "adapter not saved"

        # 3. Eval
        subprocess.run([
            sys.executable, f"{PROJECT_DIR}/evaluate.py",
            "--data_path", data_path,
            "--lora_path", lora_path,
            "--output", eval_path,
        ], check=True, timeout=300, env=_ENV)

        assert os.path.exists(eval_path), "eval report not saved"
        with open(eval_path) as f:
            report = json.load(f)
        assert "execution_accuracy" in report
        assert report["total"] == 3
        print(f"E2E passed: exec_acc={report['execution_accuracy']:.3f}")
