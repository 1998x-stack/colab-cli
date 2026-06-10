# Multi-Modal ML Tutorial

End-to-end ML tutorial covering NLP, Computer Vision, and Audio — all using HuggingFace libraries. Each section fine-tunes a pre-trained transformer on a public dataset.

## Sections

| # | Domain | Task | Dataset | Model | Classes | Samples | Epochs | ~Time |
|---|--------|------|---------|-------|---------|---------|--------|-------|
| 1 | NLP | Text classification | `ag_news` | DistilBERT | 4 topics | 2000/500 | 3 | 5 min |
| 2 | CV | Image classification | `food101` | ViT-base | 5 foods | 1250/250 | 3 | 5 min |
| 3 | Audio | Keyword spotting | `speech_commands` | Wav2Vec2 | 5 keywords | 2500/500 | 2 | 5 min |

Each section is a standalone module (`section01_nlp.py`, etc.) with a `run(output_dir) -> metrics` interface. `tutorial.py` orchestrates all three.

## Quick start

```bash
# From repo root, with proxy (required from China)
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890

# Provision, upload, launch
colab new --gpu T4 -s ml-tutorial
colab upload projects/ml-tutorial/tutorial.py /content/tutorial.py
colab upload projects/ml-tutorial/section01_nlp.py /content/section01_nlp.py
colab upload projects/ml-tutorial/section02_cv.py /content/section02_cv.py
colab upload projects/ml-tutorial/section03_audio.py /content/section03_audio.py
colab exec -s ml-tutorial -f launch.py --timeout 300

# Monitor
colab exec -s ml-tutorial -f check_progress.py --timeout 30

# Download results
echo 'import subprocess; subprocess.run(["tar","-czf","/content/results.tar.gz","-C","/content","tutorial-output"])' | colab exec -s ml-tutorial --timeout 30
colab download -s ml-tutorial /content/results.tar.gz ./results.tar.gz

# Cleanup
colab stop -s ml-tutorial
```

## Optional: HuggingFace token

For gated datasets or Hub upload, place your token at `~/.huggingface/access_token` and upload it:

```bash
mkdir -p ~/.huggingface
echo "hf_your_token_here" > ~/.huggingface/access_token
colab upload -s ml-tutorial ~/.huggingface/access_token /content/.huggingface/access_token
```

`tutorial.py` auto-detects and logs in if the token file exists.

## Output structure

```
/content/tutorial-output/
  summary.json                 # aggregate metrics across all sections
  section01_nlp/
    metrics.json, model.pt, confusion_matrix.png, sample_predictions.txt
  section02_cv/
    metrics.json, model.pt, confusion_matrix.png, sample_predictions.png
  section03_audio/
    metrics.json, model.pt, confusion_matrix.png, waveforms.png
```

## Architecture

```
tutorial.py          # orchestrator — calls each section in sequence
  ├── section01_nlp.py    # NLP: DistilBERT on AG News
  ├── section02_cv.py     # CV:  ViT on Food101
  └── section03_audio.py  # Audio: Wav2Vec2 on Speech Commands
launch.py            # Colab launcher — installs deps, spawns tutorial.py detached
check_progress.py    # Colab monitor — checks process, tails log, lists output
```

Each section module is self-contained and can run independently:

```bash
python section01_nlp.py    # runs with default output dir /tmp/nlp-test
```

## Dependencies

`launch.py` installs: `transformers`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `seaborn`, `matplotlib`, `torch`, `torchvision`.

`tutorial.py` auto-upgrades `huggingface_hub>=0.26.0` and `datasets>=3.0.0` before importing sections (fixes namespace issue on older Colab VMs).
