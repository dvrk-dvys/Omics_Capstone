"""
train_eval_job.py — Python ML training and evaluation replacing Weka.

Two stages:
  1. Baseline: all models evaluated with RepeatedStratifiedKFold, logged to MLflow
  2. Hyperopt: XGBoost + RandomForest tuned via hyperopt, logged to MLflow

All runs visible in MLflow UI at http://localhost:5001
"""

import os
import logging
import pandas as pd
import numpy as np
import mlflow
from rich.progress import track
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, Trials, STATUS_OK

# Suppress MLflow's per-run URL banners
logging.getLogger("mlflow").setLevel(logging.WARNING)

from app.models.baseline_models import BASELINE_MODELS, HYPEROPT_SPACES, make_pipeline
from app.utils.mlflow_utils import setup_mlflow, log_dataframe_artifact
from app.utils.logging_utils import get_logger

log = get_logger("train_eval_job")

SCORING = ["accuracy", "roc_auc", "f1_weighted", "balanced_accuracy"]


def prepare_data(selected_df: pd.DataFrame):
    """Encode labels and split into X, y arrays."""
    probe_cols = selected_df.columns[:-1]
    X = selected_df[probe_cols].values
    le = LabelEncoder()
    y = le.fit_transform(selected_df["class"].values)
    log.info(f"Classes: {list(le.classes_)}  (encoded 0/1)")
    return X, y, probe_cols.tolist()


def run_baseline(X: np.ndarray, y: np.ndarray, config: dict) -> pd.DataFrame:
    """Evaluate all baseline models with RepeatedStratifiedKFold."""
    cv = RepeatedStratifiedKFold(
        n_splits=config["training"]["cv_splits"],
        n_repeats=config["training"]["cv_repeats"],
        random_state=config["training"]["random_state"],
    )

    results = []
    for name, model in track(BASELINE_MODELS.items(), description="Baseline models", total=len(BASELINE_MODELS)):
        log.info(f"  {name}")
        pipeline = make_pipeline(name, model)

        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model", name)
            mlflow.log_param("cv_splits", config["training"]["cv_splits"])
            mlflow.log_param("cv_repeats", config["training"]["cv_repeats"])

            scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)

            metrics = {
                "accuracy_mean":          scores["test_accuracy"].mean(),
                "accuracy_std":           scores["test_accuracy"].std(),
                "roc_auc_mean":           scores["test_roc_auc"].mean(),
                "f1_weighted_mean":       scores["test_f1_weighted"].mean(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
            }
            mlflow.log_metrics(metrics)

            results.append({"model": name, **metrics})
            log.info(f"    acc={metrics['accuracy_mean']:.3f}  auc={metrics['roc_auc_mean']:.3f}")

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)


def run_hyperopt(X: np.ndarray, y: np.ndarray, config: dict) -> dict:
    """Run hyperopt for XGBoost and RandomForest."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["training"]["random_state"])
    best_params = {}

    for model_name in config["hyperopt"]["models"]:
        log.info(f"  Hyperopt: {model_name}")
        space = HYPEROPT_SPACES[model_name]

        def objective(params):
            if model_name == "xgboost":
                model = XGBClassifier(**params)
            else:
                model = RandomForestClassifier(**params)

            with mlflow.start_run(run_name=f"hyperopt_{model_name}", nested=True):
                mlflow.set_tag("stage", "hyperopt")
                mlflow.set_tag("model", model_name)
                mlflow.log_params(params)

                scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "balanced_accuracy"])
                auc  = scores["test_roc_auc"].mean()
                bacc = scores["test_balanced_accuracy"].mean()

                mlflow.log_metric("roc_auc_mean", auc)
                mlflow.log_metric("balanced_accuracy_mean", bacc)

            return {"loss": -auc, "status": STATUS_OK}

        with mlflow.start_run(run_name=f"hyperopt_{model_name}_search"):
            mlflow.set_tag("stage", "hyperopt_search")
            mlflow.set_tag("model", model_name)

            trials = Trials()
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=config["hyperopt"]["max_evals"],
                trials=trials,
            )
            best_params[model_name] = best
            mlflow.log_params({f"best_{k}": v for k, v in best.items()})
            log.info(f"    Best params: {best}")

    return best_params


def run(config: dict, selected_df: pd.DataFrame) -> pd.DataFrame:
    setup_mlflow(config)

    log.info("Preparing data...")
    X, y, probe_cols = prepare_data(selected_df)
    log.info(f"  X shape: {X.shape}  |  class balance: {np.bincount(y)}")

    log.info("Running baseline models...")
    comparison_df = run_baseline(X, y, config)

    log.info("\nBaseline results:")
    log.info(comparison_df[["model", "accuracy_mean", "roc_auc_mean", "f1_weighted_mean"]].to_string(index=False))

    comparison_path = config["paths"]["model_comparison"]
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    log.info(f"Saved model comparison: {comparison_path}")

    log.info("Running hyperopt...")
    best_params = run_hyperopt(X, y, config)

    return comparison_df
