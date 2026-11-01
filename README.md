# VisionGuard: AI vs Real Image Classification & Serving System

![VisionGuard CI Pipeline](https://github.com/Wahaj-07/23L-0560_A2_Ops/actions/workflows/ci.yml/badge.svg)

VisionGuard is a deep learning web service designed to classify uploaded imagery as either authentic (real) or synthesized by generative AI models. The system features disciplined experiment tracking via MLflow, containerized serving using Docker Compose with PostgreSQL audit logging, and automated testing via GitHub Actions.

## Project Structure

```text
23L-0560_A2_Ops/
├── .github/workflows/ci.yml
├── app/
│   ├── static/style.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── error.html
│   │   ├── history.html
│   │   ├── index.html
│   │   └── result.html
│   ├── __init__.py
│   ├── inference.py
│   ├── models.py
│   └── routes.py
├── data/
│   ├── raw.dvc
│   └── splits/
├── models/
│   ├── baseline_best.keras
│   ├── champion_model.keras
│   └── transfer_best.keras
├── reports/
│   ├── figures/
│   ├── dataset_audit.csv
│   └── tuning_results.csv
├── src/
│   ├── data_audit.py
│   ├── dataset.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── models.py
│   ├── prepare_splits.py
│   ├── register_champion.py
│   ├── train.py
│   └── tune.py
├── tests/
│   └── test_basic.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── run.py
```

## Setup & Running Locally

1. Create and activate Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Start application locally:
```bash
python run.py
```
Access the application at `http://localhost:5001`.

## Containerized Deployment with Docker Compose

Run the entire isolated web and PostgreSQL stack:
```bash
docker compose up --build -d
```
Check running services:
```bash
docker compose ps
```
The database is isolated inside Compose's internal network with no exposed external ports, and persistent storage is preserved in the `pgdata` volume.

