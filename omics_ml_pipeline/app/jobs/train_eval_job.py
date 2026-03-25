"""
train_eval_job.py — Python ML training and evaluation replacing Weka.

Three stages:
  1. Baseline: all models evaluated with RepeatedStratifiedKFold, logged to MLflow
  2. Hyperopt search: XGBoost + RandomForest tuned via hyperopt (nested trial runs)
  3. Tuned eval: best-params models re-evaluated with the same CV as baseline, logged to MLflow

All runs visible in MLflow UI at http://localhost:5002
"""

import io
import os
import logging
import contextlib
import pandas as pd
import numpy as np
import mlflow
from rich.progress import track
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

# Suppress MLflow's per-run URL banners
logging.getLogger("mlflow").setLevel(logging.WARNING)

from app.models.baseline_models import BASELINE_MODELS, HYPEROPT_SPACES, make_pipeline
from app.utils.mlflow_utils import setup_mlflow
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


# ---------------------------------------------------------------------------
# Shared CV evaluation helper
# ---------------------------------------------------------------------------

def _eval_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    config: dict,
    run_name: str,
    stage: str,
    model_name: str,
    source: str,
    extra_params: dict = None,
) -> dict:
    """
    Run cross-validation for one model, log everything to MLflow, return metrics dict.

    Tags logged on every run:
      stage     = baseline | tuned | hyperopt_search | hyperopt_trial | biomarker
      model     = xgboost | random_forest | mlp | ...
      run_kind  = evaluation | search | artifact_generation
      source    = baseline_default | hyperopt_tuned
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("run_kind", "evaluation")
        mlflow.set_tag("source", source)
        mlflow.log_param("cv_splits", config["training"]["cv_splits"])
        mlflow.log_param("cv_repeats", config["training"]["cv_repeats"])

        if extra_params:
            loggable = {
                k: v for k, v in extra_params.items()
                if isinstance(v, (int, float, str, bool))
            }
            mlflow.log_params(loggable)

        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        metrics = {
            "accuracy_mean":          round(float(scores["test_accuracy"].mean()), 4),
            "accuracy_std":           round(float(scores["test_accuracy"].std()),  4),
            "roc_auc_mean":           round(float(scores["test_roc_auc"].mean()),  4),
            "roc_auc_std":            round(float(scores["test_roc_auc"].std()),   4),
            "f1_weighted_mean":       round(float(scores["test_f1_weighted"].mean()), 4),
            "f1_weighted_std":        round(float(scores["test_f1_weighted"].std()),  4),
            "balanced_accuracy_mean": round(float(scores["test_balanced_accuracy"].mean()), 4),
            "balanced_accuracy_std":  round(float(scores["test_balanced_accuracy"].std()),  4),
        }
        mlflow.log_metrics(metrics)

    return metrics


# ---------------------------------------------------------------------------
# Stage 1 — Baseline
# ---------------------------------------------------------------------------

def run_baseline(X: np.ndarray, y: np.ndarray, config: dict, run_id: str = "") -> pd.DataFrame:
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
        metrics = _eval_cv(
            pipeline, X, y, cv, config,
            run_name=f"{run_id}baseline_{name}",
            stage="baseline",
            model_name=name,
            source="baseline_default",
        )
        results.append({"model": f"baseline_{name}", "source": "baseline_default", **metrics})
        log.info(f"    acc={metrics['accuracy_mean']:.3f}  auc={metrics['roc_auc_mean']:.3f}")

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)


# ---------------------------------------------------------------------------
# Stage 2 — Hyperopt search
# ---------------------------------------------------------------------------

def run_hyperopt(X: np.ndarray, y: np.ndarray, config: dict, run_id: str = "") -> dict:
    """
    Run hyperopt search for XGBoost and RandomForest.

    Trial runs are nested children of the search parent run.
    Returns best_params dict keyed by model name; values are raw fmin output
    (hp label keys) suitable for passing to space_eval().
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["training"]["random_state"])
    best_params = {}

    for model_name in config["hyperopt"]["models"]:
        log.info(f"  Hyperopt search: {model_name}")
        space = HYPEROPT_SPACES[model_name]

        def objective(params):
            if model_name == "xgboost":
                model = XGBClassifier(**params)
            else:
                model = RandomForestClassifier(**params)

            with mlflow.start_run(run_name=f"{run_id}hyperopt_{model_name}_trial", nested=True):
                mlflow.set_tag("stage", "hyperopt_trial")
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("run_kind", "search")
                loggable = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params(loggable)

                scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "balanced_accuracy"])
                auc  = scores["test_roc_auc"].mean()
                bacc = scores["test_balanced_accuracy"].mean()
                mlflow.log_metric("roc_auc_mean", auc)
                mlflow.log_metric("balanced_accuracy_mean", bacc)

            return {"loss": -auc, "status": STATUS_OK}

        with mlflow.start_run(run_name=f"{run_id}hyperopt_{model_name}_search"):
            mlflow.set_tag("stage", "hyperopt_search")
            mlflow.set_tag("model", model_name)
            mlflow.set_tag("run_kind", "search")

            trials = Trials()
            with contextlib.redirect_stdout(io.StringIO()):
                best = fmin(
                    fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=config["hyperopt"]["max_evals"],
                    trials=trials,
                    verbose=False,
                )
            best_params[model_name] = best
            mlflow.log_params({f"best_{k}": v for k, v in best.items()})
            log.info(f"    Best params: {best}")

    return best_params


# ---------------------------------------------------------------------------
# Stage 3 — Tuned evaluation
# ---------------------------------------------------------------------------

def run_tuned_models(X: np.ndarray, y: np.ndarray, config: dict, best_params: dict, run_id: str = "") -> pd.DataFrame:
    """
    Re-evaluate optimised models using best hyperopt params with the same
    RepeatedStratifiedKFold as baseline, so results are directly comparable.

    Uses space_eval() to resolve hp label names back to actual constructor kwargs.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=config["training"]["cv_splits"],
        n_repeats=config["training"]["cv_repeats"],
        random_state=config["training"]["random_state"],
    )

    results = []
    for model_name in track(config["hyperopt"]["models"], description="Tuned models"):
        log.info(f"  {model_name}")

        # Resolve hp label names → actual constructor kwargs (handles hp.choice indices, scope.int, etc.)
        params = space_eval(HYPEROPT_SPACES[model_name], best_params[model_name])

        if model_name == "xgboost":
            model = XGBClassifier(**params)
        else:
            model = RandomForestClassifier(**params)

        metrics = _eval_cv(
            model, X, y, cv, config,
            run_name=f"{run_id}tuned_{model_name}",
            stage="tuned",
            model_name=model_name,
            source="hyperopt_tuned",
            extra_params=params,
        )
        results.append({"model": f"tuned_{model_name}", "source": "hyperopt_tuned", **metrics})
        log.info(f"    acc={metrics['accuracy_mean']:.3f}  auc={metrics['roc_auc_mean']:.3f}")

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(config: dict, selected_df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
    setup_mlflow(config)

    log.info("Preparing data...")
    X, y, probe_cols = prepare_data(selected_df)
    log.info(f"  X shape: {X.shape}  |  class balance: {np.bincount(y)}")

    log.info("Running baseline models...")
    baseline_df = run_baseline(X, y, config, run_id)
    log.info("\nBaseline results:")
    log.info(baseline_df[["model", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]].to_string(index=False))

    log.info("Running hyperopt search...")
    best_params = run_hyperopt(X, y, config, run_id)

    log.info("Running tuned model evaluation...")
    tuned_df = run_tuned_models(X, y, config, best_params, run_id)
    log.info("\nTuned results:")
    log.info(tuned_df[["model", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]].to_string(index=False))

    # Combined comparison: baseline + tuned, sorted by AUC descending
    combined_df = (
        pd.concat([baseline_df, tuned_df], ignore_index=True)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    log.info("\nFull comparison (baseline + tuned):")
    log.info(
        combined_df[["model", "source", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]]
        .to_string(index=False)
    )

    comparison_path = config["paths"]["model_comparison"]
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    combined_df.to_csv(comparison_path, index=False)
    log.info(f"Saved combined comparison: {comparison_path}")

    return combined_df
