# Architecture Diagram — Heart Disease MLOps Pipeline

BITS Pilani · AIMLCZG523 · Assignment 01

```
                         ┌──────────────────────┐
                         │  UCI Heart Disease    │
                         │  Dataset (ucimlrepo)  │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ data/download_data.py │  raw CSV + data_info.txt
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ src/data_processing.py│  clean + encode + build_preprocessor
                         └───────────┬──────────┘
                                     │  data/processed/heart_disease_clean.csv
                                     ▼
                    ┌───────────────────────────────┐
                    │        src/train.py           │
                    │  3 models (LogReg / RF / XGB) │───► MLflow Tracking (mlruns/)
                    │  tuning · eval · comparison   │───► MLflow Model Registry
                    └───────────────┬───────────────┘        (HeartDiseaseClassifier
                                    │                          → Production)
                                    ▼
                    ┌───────────────────────────────┐
                    │ models/best_model_pipeline.pkl │  (preprocessor + model)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   src/api.py  (FastAPI)       │
                    │   /predict  /health  /metrics │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Docker image           │◄──── GitHub Actions CI/CD
                    │   heart-disease-api:v1        │      (lint → test → build)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Kubernetes (Minikube)       │
                    │   Deployment (2 replicas)     │
                    │   NodePort Service            │
                    └───────────────┬───────────────┘
                                    │  /metrics scraped
                                    ▼
                    ┌──────────────┐      ┌──────────────┐
                    │  Prometheus  │─────►│   Grafana    │
                    │  (scrape)    │      │  (dashboard) │
                    └──────────────┘      └──────────────┘
```

## Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| `download_data.py` | Fetch raw dataset, report missing values |
| `data_processing.py` | Impute, binarize target, build `ColumnTransformer` |
| `train.py` | Train/tune 3 models, select best by ROC-AUC, log to MLflow |
| `predict.py` | Load pipeline, serve single/batch predictions |
| `api.py` | FastAPI service + Prometheus instrumentation |
| Docker | Reproducible runtime for the API |
| Kubernetes | Scalable deployment with health probes |
| Prometheus / Grafana | Metrics collection and visualisation |
| GitHub Actions | Lint, test, and Docker-build gates |
