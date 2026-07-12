"""Train, tune, evaluate, and compare three classifiers for heart disease risk.

Each candidate is a full sklearn ``Pipeline`` that bundles the fitted
``ColumnTransformer`` preprocessor with the estimator, so the winning artifact
saved to ``models/best_model_pipeline.pkl`` is fully self-contained and can be
loaded at inference time without any external preprocessing code.

Run from the project root:
    python src/train.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    build_preprocessor,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PROCESSED_PATH = "data/processed/heart_disease_clean.csv"
MODEL_PATH = "models/best_model_pipeline.pkl"
MODEL_INFO_PATH = "models/model_info.json"
PLOTS_DIR = "screenshots/eda"
RANDOM_STATE = 42
CV_FOLDS = 5

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "heart-disease-classifier"
REGISTERED_MODEL_NAME = "HeartDiseaseClassifier"
AUTHOR = "Mohanraj G"

# Metrics we log to MLflow for every run.
METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_mean", "cv_std"]

FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_xy() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed data and split into feature matrix X and target y."""
    df = pd.read_csv(PROCESSED_PATH)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ----------------------------------------------------------------------------
# Model / search-space definitions
# ----------------------------------------------------------------------------
def build_search_spaces(
    preprocessor,
) -> Dict[str, Tuple[object, dict, str]]:
    """Return a mapping: model_name -> (search_cv, fitted_after, tuner_kind).

    Each search wraps a full Pipeline(preprocessor + model). Prefixing the
    hyperparameters with ``model__`` targets the estimator step.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- Model 1: Logistic Regression (exhaustive grid search) ---------------
    logreg_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    # Split grids keep penalty/solver combinations valid.
    logreg_grid = [
        {
            "model__solver": ["liblinear"],
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
    ]
    logreg_search = GridSearchCV(
        logreg_pipe, logreg_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    # --- Model 2: Random Forest (randomized search) --------------------------
    rf_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )
    rf_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 4, 6, 8, 12],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe,
        rf_dist,
        n_iter=15,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # --- Model 3: XGBoost (randomized search) --------------------------------
    xgb_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        xgb_dist,
        n_iter=15,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {
        "LogisticRegression": (logreg_search, {}, "GridSearchCV"),
        "RandomForest": (rf_search, {}, "RandomizedSearchCV"),
        "XGBoost": (xgb_search, {}, "RandomizedSearchCV"),
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute test-set metrics plus a 5-fold CV summary for one pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    }


def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the model comparison table (4 decimal places)."""
    header = (
        f"| {'Model':<20} | {'Accuracy':>8} | {'Precision':>9} | "
        f"{'Recall':>7} | {'F1':>7} | {'ROC-AUC':>8} | {'CV Mean':>8} |"
    )
    sep = "|" + "-" * (len(header) - 2) + "|"
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"| {name:<20} | {m['accuracy']:>8.4f} | {m['precision']:>9.4f} | "
            f"{m['recall']:>7.4f} | {m['f1_score']:>7.4f} | "
            f"{m['roc_auc']:>8.4f} | {m['cv_mean']:>8.4f} |"
        )
    print("=" * len(header))


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_roc_curves(
    fitted: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Overlay ROC curves for all models on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = {"LogisticRegression": "#2196F3", "RandomForest": "#4CAF50", "XGBoost": "#F44336"}
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=palette.get(name), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Heart Disease Classifiers")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {out}")


def plot_confusion_matrices(
    fitted: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Render one confusion matrix subplot per model."""
    n = len(fitted)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(name)
    fig.suptitle("Confusion Matrices (Test Set)")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {out}")


def plot_feature_importance(fitted: Dict[str, Pipeline]) -> None:
    """Bar chart of feature importances for the tree-based models."""
    tree_models = [n for n in ("RandomForest", "XGBoost") if n in fitted]
    if not tree_models:
        return
    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]
    for ax, name in zip(axes, tree_models):
        pipe = fitted[name]
        feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        order = np.argsort(importances)[::-1][:15]
        ax.barh(
            [feat_names[i] for i in order][::-1],
            importances[order][::-1],
            color="#673AB7",
        )
        ax.set_title(f"{name} — Top Feature Importances")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {out}")


# ----------------------------------------------------------------------------
# MLflow experiment tracking
# ----------------------------------------------------------------------------
def log_experiments(
    fitted: Dict[str, Pipeline],
    results: Dict[str, Dict[str, float]],
    best_params_all: Dict[str, dict],
    best_name: str,
) -> str:
    """Log one MLflow run per model and return the best model's run_id.

    Each run records its tuned hyperparameters, all evaluation metrics, the
    shared comparison plots, and the full fitted pipeline. The winning run is
    additionally tagged as the production candidate.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    shared_artifacts = [
        os.path.join(PLOTS_DIR, "roc_curves.png"),
        os.path.join(PLOTS_DIR, "confusion_matrices.png"),
        os.path.join(PLOTS_DIR, "feature_importance.png"),
    ]

    best_run_id = ""
    for name, pipe in fitted.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(best_params_all[name])
            for key in METRIC_KEYS:
                mlflow.log_metric(key, results[name][key])
            for artifact in shared_artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact, artifact_path="plots")
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if name == best_name:
                mlflow.set_tag("status", "production-candidate")
                mlflow.set_tag("dataset", "heart-disease-uci")
                mlflow.set_tag("author", AUTHOR)
                best_run_id = run.info.run_id
            print(f"    logged MLflow run for {name} (run_id={run.info.run_id})")

    return best_run_id


def register_best_model(best_run_id: str) -> None:
    """Register the winning run's model and promote it to the Production stage."""
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"Registered '{REGISTERED_MODEL_NAME}' v{result.version} "
        "and transitioned to stage 'Production'."
    )


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def save_best(
    best_name: str,
    best_pipe: Pipeline,
    best_params: dict,
    metrics: Dict[str, float],
) -> None:
    """Persist the winning pipeline (joblib) and its metadata (JSON)."""
    import joblib

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)

    info = {
        "model_name": best_name,
        "best_params": best_params,
        "test_metrics": metrics,
        "selected_reason": (
            f"{best_name} achieved the highest test ROC-AUC ({metrics['roc_auc']}) "
            "among the three candidates and was selected as the production model."
        ),
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nBest model saved   -> {MODEL_PATH}")
    print(f"Model metadata     -> {MODEL_INFO_PATH}")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def main() -> None:
    # Headless backend — save figures without needing a display (CI-safe).
    plt.switch_backend("Agg")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading processed data...")
    X, y = load_xy()
    print(f"Feature matrix: {X.shape}  |  Target balance: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    preprocessor = build_preprocessor(X)
    searches = build_search_spaces(preprocessor)

    fitted: Dict[str, Pipeline] = {}
    best_params_all: Dict[str, dict] = {}
    results: Dict[str, Dict[str, float]] = {}

    for name, (search, _, tuner) in searches.items():
        print(f"\n>>> Tuning {name} via {tuner} ...")
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        fitted[name] = best_pipe
        # Strip the 'model__' prefix for cleaner reporting/logging.
        best_params_all[name] = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        results[name] = evaluate(best_pipe, X_train, y_train, X_test, y_test)
        print(f"    best params: {best_params_all[name]}")
        print(f"    test ROC-AUC: {results[name]['roc_auc']}")

    print_comparison(results)

    # Save comparison plots.
    plot_roc_curves(fitted, X_test, y_test)
    plot_confusion_matrices(fitted, X_test, y_test)
    plot_feature_importance(fitted)

    # Select winner by ROC-AUC.
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nWinner: {best_name} (ROC-AUC = {results[best_name]['roc_auc']})")
    save_best(best_name, fitted[best_name], best_params_all[best_name], results[best_name])

    # MLflow experiment tracking + model registry.
    print("\nLogging experiments to MLflow...")
    best_run_id = log_experiments(fitted, results, best_params_all, best_name)
    if best_run_id:
        register_best_model(best_run_id)

    print(
        "\nACTION REQUIRED: Launch the MLflow UI and take screenshots -> screenshots/mlflow/\n"
        "    mlflow ui --backend-store-uri ./mlruns --port 5000\n"
        "  Then open http://localhost:5000 and capture:\n"
        "    1. Experiments list view\n"
        "    2. Run comparison table (select all 3 runs -> Compare)\n"
        "    3. Best model artifact browser\n"
        "    4. Model Registry -> HeartDiseaseClassifier -> Production stage"
    )


if __name__ == "__main__":
    # Allow running via `python src/train.py` from the project root.
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
    main()
