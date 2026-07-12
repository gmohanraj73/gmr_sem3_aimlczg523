# Heart Disease Risk Predictor — MLOps Pipeline

![CI](https://github.com/USERNAME/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

> BITS Pilani · AIMLCZG523 · MLOps Assignment 01

A complete, production-grade MLOps pipeline that trains a heart-disease risk
classifier on the UCI Heart Disease dataset and serves it as a cloud-ready,
monitored REST API.

## Project Overview

The project covers the full ML lifecycle: data acquisition and cleaning,
exploratory analysis, model development with hyperparameter tuning, experiment
tracking and model registry (MLflow), automated testing and CI/CD (GitHub
Actions), containerisation (Docker), orchestration (Kubernetes/Minikube), and
observability (Prometheus + Grafana). The winning model is packaged as a single
self-contained sklearn `Pipeline` (preprocessor + estimator) so inference needs
no external preprocessing code.

## Architecture

```
[UCI Dataset] --> download_data.py --> data_processing.py --> train.py + MLflow
                                                                   |
                                          models/best_model_pipeline.pkl
                                                                   |
                                                      api.py (FastAPI) --> Docker
                                                       |                     |
                                              MLflow Registry      Kubernetes (Minikube)
                                                                             |
                                                        Prometheus --> Grafana dashboard
                                                              ^
                                                     GitHub Actions CI/CD
```

See [reports/architecture_diagram.md](reports/architecture_diagram.md) for the
detailed diagram.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| Data / ML | pandas, numpy, scikit-learn, XGBoost |
| Experiment tracking | MLflow (tracking + model registry) |
| API | FastAPI, Uvicorn, Pydantic |
| Testing / lint | pytest, pytest-cov, flake8, ruff, black |
| CI/CD | GitHub Actions |
| Container | Docker |
| Orchestration | Kubernetes (Minikube) |
| Monitoring | Prometheus, Grafana, prometheus-fastapi-instrumentator |

## Quick Start

### Prerequisites
- Python 3.10, Git
- (Optional) Docker, Minikube + kubectl for deployment phases

### Installation

**Option A — pip + venv**
```bash
git clone <repo-url>
cd heart-disease-mlops
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt -r requirements-dev.txt
```

**Option B — conda**
```bash
conda env create -f environment.yml
conda activate mlops-heart-disease
```

### Run Training
```bash
python data/download_data.py     # -> data/raw/heart_disease_raw.csv
python src/data_processing.py    # -> data/processed/heart_disease_clean.csv
python -m src.train              # trains 3 models, logs to MLflow, saves winner
```

### Run MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

### Run API Locally
```bash
uvicorn src.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Run Docker
```bash
docker build -t heart-disease-api:v1 .
docker run -d -p 8000:8000 --name heart-api heart-disease-api:v1
curl http://localhost:8000/health
```

### Deploy to Kubernetes (Minikube)
```bash
minikube start --memory=4096 --cpus=2
# Edit k8s/deployment.yaml -> set 'image:' to your pushed image
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -w
minikube service heart-disease-api-service --url
```

### Monitoring stack
```bash
cd docker
docker-compose -f docker-compose.monitoring.yml up -d
# API:        http://localhost:8000/health
# Prometheus: http://localhost:9090/targets
# Grafana:    http://localhost:3000  (admin / admin123)
python ../scripts/load_test.py     # generate traffic for the dashboards
```

## API Reference

Feature encoding follows the **original UCI Heart Disease** dataset (as returned
by `ucimlrepo` id=45), which is what the model is trained on.

### `POST /predict`
Request body (all fields required):
```json
{
  "age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 229,
  "fbs": 0, "restecg": 2, "thalach": 129, "exang": 1, "oldpeak": 2.6,
  "slope": 2, "ca": 2, "thal": 7
}
```
Response:
```json
{
  "prediction": 1, "probability": 0.9865, "risk_level": "High",
  "model_version": "1.0.0", "latency_ms": 31.07
}
```
Field ranges: `cp` 1–4, `restecg` 0–2, `slope` 1–3, `ca` 0–3,
`thal` ∈ {3,6,7}. Invalid or missing fields return HTTP 422.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":120,"chol":229,"fbs":0,"restecg":2,"thalach":129,"exang":1,"oldpeak":2.6,"slope":2,"ca":2,"thal":7}'
```

### `GET /health`
Returns `{"status": "healthy", "model_loaded": true}`.

### `GET /metrics`
Prometheus-format metrics (request counts, latency histograms, in-progress gauge).

## Project Structure

```
heart-disease-mlops/
├── data/               raw + processed datasets, download_data.py
├── notebooks/          01_eda.ipynb, 02_training.ipynb
├── src/                data_processing.py, train.py, predict.py, api.py
├── tests/              test_data.py, test_model.py, test_api.py
├── models/             best_model_pipeline.pkl, model_info.json
├── mlruns/             MLflow tracking store
├── scripts/            test_inference.py, load_test.py
├── docker/             docker-compose.monitoring.yml, prometheus.yml
├── k8s/                deployment.yaml, service.yaml
├── screenshots/        eda / mlflow / cicd / docker / k8s / monitoring
├── reports/            architecture_diagram.md, REPORT_OUTLINE.md
├── .github/workflows/  ci.yml
├── Dockerfile
├── requirements.txt / requirements-dev.txt / environment.yml
└── README.md
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean |
|-------|----------|-----------|--------|------|---------|---------|
| **Logistic Regression** ⭐ | 0.8689 | 0.8333 | 0.8929 | 0.8621 | **0.9589** | 0.9070 |
| Random Forest | 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9481 | 0.9017 |
| XGBoost | 0.8852 | 0.8621 | 0.8929 | 0.8772 | 0.9437 | 0.8846 |

The winner is selected by test-set ROC-AUC. Full justification in
`notebooks/02_training.ipynb`.

## CI/CD Pipeline

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs three
gated jobs on push/PR to `main`/`dev`:
1. **lint** — `flake8` + `ruff` (fails loudly on any violation)
2. **test** — installs deps, downloads data, trains the model, runs `pytest` with coverage
3. **build-docker** — builds the image and smoke-tests `/health` in a running container

## Monitoring Setup

The API auto-exposes Prometheus metrics at `/metrics` via
`prometheus-fastapi-instrumentator`. `docker/docker-compose.monitoring.yml`
brings up the API alongside Prometheus (scraping `api:8000`) and Grafana. Import
Grafana dashboard **12708** (FastAPI Observability) or build panels for request
rate, error rate, P95 latency, and in-progress requests.

## Screenshots

Evidence for each phase lives under `screenshots/` (`eda`, `mlflow`, `cicd`,
`docker`, `k8s`, `monitoring`).

## License

MIT (academic use — AIMLCZG523 Assignment 01).
