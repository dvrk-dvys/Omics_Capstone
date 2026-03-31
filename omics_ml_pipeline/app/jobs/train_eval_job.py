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
import time
import logging
import contextlib
import pandas as pd
import numpy as np
import mlflow
from rich.progress import track, Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

# Suppress MLflow's per-run URL banners
logging.getLogger("mlflow").setLevel(logging.WARNING)

from app.models.baseline_models import get_baseline_models, get_hyperopt_spaces, make_pipeline
from app.utils.feature_select import (
    plot_roc_curves, plot_confusion_matrix, plot_feature_importance,
    plot_model_comparison_bar, plot_composite_eval,
    plot_statistical_vs_model_importance,
    plot_gene_importance_aggregated, plot_biomarker_summary_composite,
)
from app.utils.mlflow_utils import setup_mlflow
from app.utils.logging_utils import get_logger, console

log = get_logger("train_eval_job")

SCORING = ["accuracy", "roc_auc", "f1_weighted", "balanced_accuracy"]


def prepare_data(selected_df: pd.DataFrame):
    """Encode labels and split into X, y arrays."""
    probe_cols = selected_df.columns[:-1]
    X = selected_df[probe_cols].values
    le = LabelEncoder()
    y = le.fit_transform(selected_df["class"].values)
    log.info(f"🏷️  Classes: {list(le.classes_)}  (encoded 0/1)")
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

        t0 = time.perf_counter()
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        duration_s = round(time.perf_counter() - t0, 2)

        metrics = {
            "accuracy_mean":          round(float(scores["test_accuracy"].mean()), 4),
            "accuracy_std":           round(float(scores["test_accuracy"].std()),  4),
            "roc_auc_mean":           round(float(scores["test_roc_auc"].mean()),  4),
            "roc_auc_std":            round(float(scores["test_roc_auc"].std()),   4),
            "f1_weighted_mean":       round(float(scores["test_f1_weighted"].mean()), 4),
            "f1_weighted_std":        round(float(scores["test_f1_weighted"].std()),  4),
            "balanced_accuracy_mean": round(float(scores["test_balanced_accuracy"].mean()), 4),
            "balanced_accuracy_std":  round(float(scores["test_balanced_accuracy"].std()),  4),
            "duration_s":             duration_s,
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

    project = config.get("project", {})
    scale_pos_weight = project.get("n_disease", 30) / project.get("n_control", 10)
    baseline_models = get_baseline_models(scale_pos_weight)

    results = []
    for name, model in track(baseline_models.items(), description="Baseline models", total=len(baseline_models), console=console):
        log.info(f"  🤖 {name}")
        pipeline = make_pipeline(name, model)
        metrics = _eval_cv(
            pipeline, X, y, cv, config,
            run_name=f"{run_id}baseline_{name}",
            stage="baseline",
            model_name=name,
            source="baseline_default",
        )
        results.append({"model": f"baseline_{name}", "source": "baseline_default", **metrics})
        log.info(f"    📈 acc={metrics['accuracy_mean']:.3f}  auc={metrics['roc_auc_mean']:.3f}")

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

    project = config.get("project", {})
    scale_pos_weight = project.get("n_disease", 30) / project.get("n_control", 10)
    hyperopt_spaces = get_hyperopt_spaces(scale_pos_weight)

    for model_name in config["hyperopt"]["models"]:
        log.info(f"  🔍 Hyperopt search: {model_name}")
        space = hyperopt_spaces[model_name]
        max_evals = config["hyperopt"]["max_evals"]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Hyperopt {model_name}", total=max_evals)

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

                progress.advance(task)
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
                        max_evals=max_evals,
                        trials=trials,
                        verbose=False,
                    )
                best_params[model_name] = best
                mlflow.log_params({f"best_{k}": v for k, v in best.items()})
                log.info(f"    ✅ Best params: {best}")

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

    project = config.get("project", {})
    scale_pos_weight = project.get("n_disease", 30) / project.get("n_control", 10)
    hyperopt_spaces = get_hyperopt_spaces(scale_pos_weight)

    results = []
    for model_name in track(config["hyperopt"]["models"], description="Tuned models", console=console):
        log.info(f"  🤖 {model_name}")

        # Resolve hp label names → actual constructor kwargs (handles hp.choice indices, scope.int, etc.)
        params = space_eval(hyperopt_spaces[model_name], best_params[model_name])

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
        log.info(f"    📈 acc={metrics['accuracy_mean']:.3f}  auc={metrics['roc_auc_mean']:.3f}")

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(config: dict, selected_df: pd.DataFrame, run_id: str = "", gene_map: pd.Series = None) -> pd.DataFrame:
    setup_mlflow(config)

    project = config.get("project", {})
    scale_pos_weight = project.get("n_disease", 30) / project.get("n_control", 10)

    log.info("🔧 Preparing data...")
    X, y, probe_cols = prepare_data(selected_df)
    log.info(f"  📐 X shape: {X.shape}  |  class balance: {np.bincount(y)}")

    log.info("🤖 Running baseline models...")
    baseline_df = run_baseline(X, y, config, run_id)
    log.info("\n📊 Baseline results:")
    log.info(baseline_df[["model", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]].to_string(index=False))

    log.info("🔍 Running hyperopt search...")
    best_params = run_hyperopt(X, y, config, run_id)

    log.info("⚡ Running tuned model evaluation...")
    tuned_df = run_tuned_models(X, y, config, best_params, run_id)
    log.info("\n📊 Tuned results:")
    log.info(tuned_df[["model", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]].to_string(index=False))

    # Combined comparison: baseline + tuned, sorted by AUC descending
    combined_df = (
        pd.concat([baseline_df, tuned_df], ignore_index=True)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    log.info("\n🏆 Full comparison (baseline + tuned):")
    log.info(
        combined_df[["model", "source", "roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]]
        .to_string(index=False)
    )

    comparison_path = config["paths"]["model_comparison"]
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    combined_df.to_csv(comparison_path, index=False)
    log.info(f"💾 Saved combined comparison: {comparison_path}")

    # --- ML evaluation plots -------------------------------------------------
    plots_dir   = config["paths"]["plots_dir"]
    top_n       = config["feature_selection"]["top_n_feats"]
    mode        = config.get("_mode", "")
    mode_suffix = f"_{mode}_top{top_n}" if mode else f"_top{top_n}"
    mode_label  = f"  [mode={mode}, top_n={top_n}]" if mode else f"  [top_n={top_n}]"
    dataset     = f"{config.get('project', {}).get('dataset', '')} — top {top_n} probes{mode_label}"
    os.makedirs(plots_dir, exist_ok=True)

    # Reconstruct tuned XGBoost with resolved best params for plotting
    hyperopt_spaces_plot = get_hyperopt_spaces(scale_pos_weight)
    tuned_xgb_params     = space_eval(hyperopt_spaces_plot["xgboost"], best_params["xgboost"])
    tuned_xgb            = XGBClassifier(**tuned_xgb_params)

    # Baseline logistic for ROC comparison
    baseline_models_plot = get_baseline_models(scale_pos_weight)
    log_pipe             = make_pipeline("logistic_elasticnet", baseline_models_plot["logistic_elasticnet"])

    named_models = [
        ("Tuned XGBoost",       tuned_xgb),
        ("Logistic ElasticNet", log_pipe),
    ]

    log.info("📊 Generating ML evaluation plots...")

    plot_roc_curves(
        X, y, named_models, plots_dir,
        cv_splits    = config["training"]["cv_splits"],
        cv_repeats   = config["training"]["cv_repeats"],
        random_state = config["training"]["random_state"],
        dataset      = dataset,
    )

    # class_names sorted to match LabelEncoder (np.unique order)
    class_names = sorted(selected_df["class"].unique().tolist())
    plot_confusion_matrix(
        X, y, tuned_xgb, "Tuned XGBoost", class_names, plots_dir,
        cv_splits    = config["training"]["cv_splits"],
        random_state = config["training"]["random_state"],
        dataset      = dataset,
    )

    plot_feature_importance(
        X, y, probe_cols, tuned_xgb, "Tuned XGBoost", gene_map, plots_dir,
        dataset = dataset,
    )

    plot_gene_importance_aggregated(
        X, y, probe_cols, tuned_xgb, "Tuned XGBoost", gene_map, plots_dir,
        dataset = dataset,
    )

    shortlist_csv = config["paths"]["biomarker_shortlist"]
    plot_biomarker_summary_composite(
        plots_dir    = plots_dir,
        shortlist_csv= shortlist_csv,
        output_path  = os.path.join(plots_dir, "fig_biomarker_summary_composite.png"),
        dataset      = dataset,
    )

    # Figure 2 — model comparison bar chart (all models)
    plot_model_comparison_bar(
        models_csv  = comparison_path,
        plots_dir   = plots_dir,
        dataset     = dataset,
        mode_suffix = mode_suffix,
    )

    # Figure 3 — final model evaluation composite (ROC | CM | feature importance)
    eval_composite_path = os.path.join(plots_dir, "fig_3_model_eval.png")
    plot_composite_eval(plots_dir, eval_composite_path, dataset=dataset)

    # Figure 4 — statistical vs model importance scatter
    # biomarker.top_n_display also routes to: biomarker_job.py (shortlist preview log)
    top_genes_csv = os.path.join(config["paths"]["feature_select_dir"], "top100_genes.csv")
    plot_statistical_vs_model_importance(
        X, y, probe_cols, tuned_xgb, gene_map,
        top_genes_csv = top_genes_csv,
        plots_dir     = plots_dir,
        dataset       = dataset,
        random_state  = config["training"]["random_state"],
        label_top_n   = config["biomarker"].get("top_n_display", 20),
        mode_suffix   = mode_suffix,
    )

    # Rename remaining mode-dependent plots to include mode+top_n suffix
    _rename_plots = [
        "confusion_matrix.png",
        "roc_curves.png",
        "feature_importance.png",
        "feature_importance_zoomed.png",
        "gene_importance_aggregated.png",
        "gene_importance_aggregated_zoomed.png",
        "fig_3_model_eval.png",
        "fig_biomarker_summary_composite.png",
    ]
    for _fname in _rename_plots:
        _old = os.path.join(plots_dir, _fname)
        _new = os.path.join(plots_dir, _fname.replace(".png", f"{mode_suffix}.png"))
        if os.path.exists(_old):
            os.rename(_old, _new)

    log.info(f"✅ ML evaluation plots saved to: {plots_dir}  (suffix: {mode_suffix})")

    return combined_df
