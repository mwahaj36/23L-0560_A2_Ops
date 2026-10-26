# 23L-0560_A2_Ops - VisionGuard

[![VisionGuard CI](https://github.com/mwahaj36/23L-0560_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/mwahaj36/23L-0560_A2_Ops/actions/workflows/ci.yml)

VisionGuard estimates whether an uploaded artwork image is AI-generated or real. It is a classifier, not an authenticity certificate. This repository is Muhammad Wahaj's Assignment 2 submission (roll number 23L-0560).

## Reproduce locally

Use Ubuntu 22.04 or WSL with Python 3.10. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data_audit.py
python src/train.py --name baseline --kind baseline --epochs 8
python src/train.py --name mobilenet_head --kind transfer --epochs 8
```

The dataset comes from [Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images). Place its extracted `AiArtData/AiArtData` and `RealArt/RealArt` folders inside `data_extracted`, or run `dvc pull` after configuring access to the DagsHub remote. Exact duplicate hashes stay within one split.

To run the UI, put the selected model at `models/champion.keras`, set the environment values from `.env.example`, and start `python src/app.py` or `docker compose up --build`. The Docker stack has a PostgreSQL service reachable only on its internal network. Set `API_URL` to the public Space URL to route local UI predictions through the cloud API. The API container itself sets `API_MODE=1` and accepts multipart uploads at `/predict`.

## Project links

- [GitHub repository](https://github.com/mwahaj36/23L-0560_A2_Ops)
- [DagsHub project](https://dagshub.com/mwahaj36/23L-0560_A2_Ops)
- [Cloud API Space](https://huggingface.co/spaces/mwahaj36/23L-0560_A2_Ops)

## Data and experiment lineage

The CSV split files in `reports/splits` are generated with seed 560. MLflow records model settings, epoch metrics, test scores, and plots. The selected Keras model is exported into `models/champion.keras` for serving; it loads once when the API process starts. `.env` holds tokens and passwords and is ignored by Git.

