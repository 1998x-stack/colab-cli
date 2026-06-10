"""Multi-Modal ML Tutorial — orchestrates NLP, CV, and Audio sections.

Usage:
  python tutorial.py [--output-dir /content/tutorial-output]

Each section is a self-contained module with a run(output_dir) -> metrics dict.
The orchestrator runs all three, collects metrics, and writes a summary report.
"""

import json, os, sys, time, subprocess
from datetime import datetime

OUTPUT_DIR = "/content/tutorial-output"


def _ensure_hf_libs():
    """Force-upgrade huggingface_hub & datasets so short names (ag_news) work."""
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--root-user-action=ignore",
        "huggingface_hub>=0.26.0", "datasets>=3.0.0", "-q",
    ])


def maybe_login():
    """Log into HuggingFace if token file exists in common locations."""
    token_paths = [
        "/content/.huggingface/access_token",
        os.path.expanduser("~/.huggingface/token"),
    ]
    token = None
    for p in token_paths:
        if os.path.exists(p):
            token = open(p).read().strip()
            break

    if token:
        from huggingface_hub import login
        login(token=token)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [TUTORIAL] HF login OK")
        return True
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [TUTORIAL] No HF token found — using public datasets only")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    args = p.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t0_total = time.time()
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] [TUTORIAL] ====== Multi-Modal ML Tutorial ======")

    # Auth
    logged_in = maybe_login()

    # Ensure HF libs are recent enough for short dataset names
    _ensure_hf_libs()

    # Section 1: NLP
    t = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{t}] [TUTORIAL] ====== Section 1/3: NLP =====")
    from section01_nlp import run as run_nlp
    metrics_nlp = run_nlp(output_dir)

    # Section 2: CV
    t = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{t}] [TUTORIAL] ====== Section 2/3: CV =====")
    from section02_cv import run as run_cv
    metrics_cv = run_cv(output_dir)

    # Section 3: Audio
    t = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{t}] [TUTORIAL] ====== Section 3/3: Audio =====")
    from section03_audio import run as run_audio
    metrics_audio = run_audio(output_dir)

    # ── Summary report ────────────────────────────────────────────
    total_time = time.time() - t0_total
    t = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{t}] [TUTORIAL] ====== Summary =====")

    summary = {
        "tutorial": "multi-modal-ml",
        "date": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 1),
        "hf_logged_in": logged_in,
        "sections": {
            "01_nlp": metrics_nlp,
            "02_cv": metrics_cv,
            "03_audio": metrics_audio,
        },
        "overall": {
            "nlp_accuracy": metrics_nlp["test_accuracy"],
            "cv_accuracy": metrics_cv["test_accuracy"],
            "audio_accuracy": metrics_audio["test_accuracy"],
            "average_accuracy": round(
                (metrics_nlp["test_accuracy"] + metrics_cv["test_accuracy"] + metrics_audio["test_accuracy"]) / 3, 4
            ),
            "total_train_time_minutes": round(total_time / 60, 1),
        },
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  NLP    accuracy: {metrics_nlp['test_accuracy']:.4f}  ({metrics_nlp['train_time_seconds']:.0f}s)")
    print(f"  CV     accuracy: {metrics_cv['test_accuracy']:.4f}  ({metrics_cv['train_time_seconds']:.0f}s)")
    print(f"  Audio  accuracy: {metrics_audio['test_accuracy']:.4f}  ({metrics_audio['train_time_seconds']:.0f}s)")
    print(f"  ─────────────────────────────────────")
    print(f"  Average accuracy: {summary['overall']['average_accuracy']:.4f}")
    print(f"  Total time: {total_time/60:.1f}m")
    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary: {output_dir}/summary.json")


if __name__ == "__main__":
    main()
