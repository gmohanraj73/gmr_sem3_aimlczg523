# Final Report Outline — Heart Disease MLOps Pipeline

BITS Pilani · AIMLCZG523 · Assignment 01 · (target ~10 pages)

Fill each page with your own narrative and the referenced screenshots from
`screenshots/`. Keep commentary personalised (see academic-integrity notes).

## Page 1 — Title & Project Overview
- Title, name, ID, course, date
- 1-paragraph problem statement (heart-disease risk classification)
- High-level architecture thumbnail (`reports/architecture_diagram.md`)

## Page 2 — Setup & Installation
- Prerequisites, pip and conda install options
- Reproducibility note (`python data/download_data.py && python -m src.train`)
- Folder-structure tree

## Page 3 — EDA Findings
- Dataset overview (303 rows, 13 features, binary target)
- Class balance (164 no-disease / 139 disease — mild imbalance)
- Key visuals: missing-value heatmap, feature distributions, correlation heatmap
- Screenshots: `screenshots/eda/`

## Page 4 — Model Development
- Feature engineering: StandardScaler (numeric) + OneHotEncoder (categorical)
- Pipeline design (preprocessor + estimator in one artifact)
- Training approach: stratified 80/20 split, GridSearchCV / RandomizedSearchCV,
  5-fold StratifiedKFold

## Page 5 — Model Comparison & Winner
- Comparison table (Accuracy / Precision / Recall / F1 / ROC-AUC / CV mean)
- ROC curves + confusion matrices (`screenshots/eda/`)
- Winner: Logistic Regression (ROC-AUC 0.9589) — justification

## Page 6 — MLflow Experiment Tracking
- Experiment + 3 runs, logged params/metrics/artifacts
- Model Registry: HeartDiseaseClassifier → Production
- Screenshots: `screenshots/mlflow/`

## Page 7 — CI/CD Pipeline
- GitHub Actions workflow (lint → test → build-docker)
- "Fail loudly" behaviour on lint/test errors
- Screenshot: passing run (`screenshots/cicd/`)

## Page 8 — Docker & API Design
- FastAPI endpoints (`/predict`, `/health`, `/metrics`), Pydantic validation
- Dockerfile highlights (layer caching, non-root user, HEALTHCHECK)
- Screenshots: build output + `/predict` response (`screenshots/docker/`)

## Page 9 — Kubernetes Deployment
- Deployment (2 replicas, resource limits, liveness/readiness probes)
- NodePort Service, `minikube service` URL
- Screenshots: `kubectl get pods`, endpoint test (`screenshots/k8s/`)

## Page 10 — Monitoring & Conclusion
- Prometheus targets + Grafana dashboard (request rate, latency, errors)
- Screenshots: `screenshots/monitoring/`
- Lessons learned, future work, repository link
