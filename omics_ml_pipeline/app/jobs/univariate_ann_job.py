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

from app.utils.univariate_ann import run_univariate_pipeline
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
      biomarker_shortlist.csv             (also at config["paths"]["univariate_biomarker_shortlist"])
      top100_features_univariate_ann.csv / .arff  (PRIMARY Weka output)
      top500_features_univariate_ann.csv / .arff  (optional exploratory output)

    Returns:
        summary_df : per-probe aggregated summary DataFrame, sorted by Median_TestAUC
    """
    ua      = config.get("univariate_ann", {})
    paths   = config["paths"]
    project = config.get("project", {})

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
        # top100 = PRIMARY Weka output (course-faithful / report-faithful)
        # top500 = optional exploratory output — not the default Weka/report path
        weka_top_ns      = [100, 500],
        relation_name    = "univariate_ann",
        log_fn           = log.info,
    )

    n_valid = int(summary_df["Median_TestAUC"].notna().sum())
    log.info(
        f"Univariate ANN complete. "
        f"{n_valid} probes with valid Median_TestAUC out of {len(summary_df)} total."
    )
    log.info(f"  Shortlist  : {paths.get('univariate_biomarker_shortlist', output_dir + '/biomarker_shortlist.csv')}")
    log.info(f"  Weka ARFF  : {output_dir}/top100_features_univariate_ann.arff  (PRIMARY)")
    log.info(f"  Weka ARFF  : {output_dir}/top500_features_univariate_ann.arff  (optional exploratory)")

    return summary_df
