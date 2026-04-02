"""
univariate_ann_job.py — App pipeline orchestrator for the professor-faithful
univariate ANN workflow.

Thin wrapper: reads all parameters from pipeline.yaml and delegates
to run_univariate_pipeline() in app/utils/univariate_ann.py.

Invoked by app/main.py via the --univariate flag.

To run the same logic standalone (Weka prep path), use:
  python -m app.utils.univariate_ann
"""

import os

import mlflow

from app.utils.univariate_ann import run_univariate_pipeline
from app.utils.mlflow_utils import setup_mlflow
from app.utils.logging_utils import get_logger

log = get_logger("univariate_ann_job")


def run(config: dict, run_id: str = "") -> "pd.DataFrame":  # noqa: F821
    """
    Run the full univariate ANN pipeline using config-driven paths and parameters.

    Reads from:
      config["paths"]["preprocessed_csv"]
      config["paths"]["soft_file"]
      config["paths"]["univariate_ann_dir"]
      config["univariate_ann"]   (all ANN hyperparameters)
      config["project"]          (disease_label, control_label)

    Writes to config["paths"]["univariate_ann_dir"]:
      filter_univariate_auc.csv
      wrapper_performance.csv
      wrapper_summary_per_gene.csv
      wrapper_predictions.csv
      ann_probe_ranking.csv               (also at config["paths"]["univariate_ann_ranking"])
      top100_features_univariate_ann.csv / .arff  (PRIMARY Weka output)
      top500_features_univariate_ann.csv / .arff  (optional exploratory output)

    Returns:
        summary_df : per-probe aggregated summary DataFrame, sorted by Median_TestAUC
    """
    setup_mlflow(config)

    ua      = config.get("univariate_ann", {})
    paths   = config["paths"]
    project = config.get("project", {})
    top_n   = config["feature_selection"]["top_n_feats"]

    output_dir = paths["univariate_ann_dir"]
    os.makedirs(output_dir, exist_ok=True)

    perf_df, summary_df, pred_df, gene_map = run_univariate_pipeline(
        preprocessed_csv = paths["preprocessed_csv"],
        soft_gz_path     = paths["soft_file"],
        output_dir       = output_dir,
        disease_label    = project.get("disease_label", "SONFH"),
        control_label    = project.get("control_label", "control"),
        model_type       = ua.get("model_type",        "professor_ann"),
        use_filter       = ua.get("use_filter",        True),
        filter_top_n     = ua.get("filter_top_n",      2000),
        n_mccv           = ua.get("n_mccv",            5),
        train_frac       = ua.get("train_frac",        0.70),
        seed             = ua.get("seed",              123),
        epochs           = ua.get("epochs",            60),
        batch_size       = ua.get("batch_size",        16),
        patience         = ua.get("patience",          8),
        lr               = ua.get("learning_rate",     0.001),
        val_frac         = ua.get("val_frac",          0.20),
        min_train        = ua.get("min_train_samples", 10),
        min_test         = ua.get("min_test_samples",  5),
        max_na_fraction  = ua.get("max_na_fraction",   0.20),
        # top_n driven by config["feature_selection"]["top_n_feats"]
        weka_top_ns      = [top_n],
        relation_name    = "univariate_ann",
        log_fn           = log.info,
    )

    n_valid = int(summary_df["Median_TestAUC"].notna().sum())
    log.info(
        f"Univariate ANN complete. "
        f"{n_valid} probes with valid Median_TestAUC out of {len(summary_df)} total."
    )
    log.info(f"  ANN ranking : {paths.get('univariate_ann_ranking', output_dir + '/ann_probe_ranking.csv')}")
    log.info(f"  CSV (top-{top_n}): {output_dir}/top{top_n}_features_univariate_ann.csv")
    log.info(f"  ARFF(top-{top_n}): {output_dir}/top{top_n}_features_univariate_ann.arff")

    # --- MLflow logging — one run summarising the full univariate screening ---
    import pandas as pd

    model_type = ua.get("model_type", "professor_ann")
    valid = (
        summary_df[summary_df["Median_TestAUC"].notna()]
        .sort_values("Median_TestAUC", ascending=False)
        .copy()
    )
    top1 = valid.iloc[0] if not valid.empty else None

    with mlflow.start_run(run_name=f"{run_id}univariate_ann_{model_type}"):
        mlflow.set_tag("stage",    "univariate_ann")
        mlflow.set_tag("model",    model_type)
        mlflow.set_tag("run_kind", "evaluation")
        mlflow.set_tag("dataset",  config.get("project", {}).get("dataset", ""))
        if top1 is not None:
            mlflow.set_tag("top1_probe", str(top1["Gene"]))
            top1_gene = gene_map.get(str(top1["Gene"]), "---") if gene_map is not None and len(gene_map) else "---"
            mlflow.set_tag("top1_gene", top1_gene)

        mlflow.log_params({
            "model_type":    model_type,
            "n_mccv":        ua.get("n_mccv",            5),
            "filter_top_n":  ua.get("filter_top_n",      2000),
            "train_frac":    ua.get("train_frac",        0.70),
            "epochs":        ua.get("epochs",            60),
            "batch_size":    ua.get("batch_size",        16),
            "patience":      ua.get("patience",          8),
            "learning_rate": ua.get("learning_rate",     0.001),
            "val_frac":      ua.get("val_frac",          0.20),
            "seed":          ua.get("seed",              123),
        })

        top10_auc = float(valid.head(10)["Median_TestAUC"].median()) if not valid.empty else 0.0
        mlflow.log_metrics({
            "n_probes_tested":     len(summary_df),
            "n_probes_valid":      n_valid,
            "top1_median_auc":     round(float(top1["Median_TestAUC"]), 4) if top1 is not None else 0.0,
            "top1_sd_auc":         round(float(top1["SD_TestAUC"]), 4) if top1 is not None and pd.notna(top1["SD_TestAUC"]) else 0.0,
            "top10_median_auc":    round(top10_auc, 4),
            "n_probes_auc_gt_0_9": int((valid["Median_TestAUC"] > 0.9).sum()),
        })

        summary_path = os.path.join(output_dir, "wrapper_summary_per_gene.csv")
        png_path     = os.path.join(output_dir, f"univariate_probe_auc_distribution_top{top_n}.png")
        for artifact_path in (summary_path, png_path):
            if os.path.exists(artifact_path):
                try:
                    mlflow.log_artifact(artifact_path)
                except Exception as art_err:
                    log.warning(f"MLflow artifact upload skipped ({os.path.basename(artifact_path)}): {art_err}")

    log.info(f"MLflow run logged: stage=univariate_ann  model={model_type}")

    return summary_df
