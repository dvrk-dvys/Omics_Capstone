"""
feature_select_job.py — Orchestrates feature selection with config-driven paths.

Imports primitive functions from feature_select.py directly rather than
calling select_top_probes(), which has a hardcoded SOFT_GZ path internally.

Produces the same outputs as the standalone feature_select.py Weka path:
  - top50_features.csv
  - gene_rankings.csv
  - gene_level_summary.csv
  - EDA plots: volcano, fold-change bar, box plots, sample correlation, heatmap, PCA
"""

import os
import pandas as pd
from rich.progress import track

from app.utils.logging_utils import get_logger, console
from app.utils.feature_select import (
    load_preprocessed,
    rank_by_fold_change,
    rank_by_variance,
    load_probe_annotation,
    build_gene_level_summary,
    plot_volcano,
    plot_fold_change_bar,
    plot_boxplots,
    plot_sample_correlation,
    plot_heatmap,
    plot_pca,
)
log = get_logger("feature_select_job")


def run(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
        selected_df  : DataFrame — top N probes + class column
        fc_ranking   : Series   — full fold-change ranking (all probes)
        gene_map     : Series   — probe_id → gene_symbol
    """
    input_path    = config["paths"]["preprocessed_csv"]
    soft_path     = config["paths"]["soft_file"]
    output_dir    = config["paths"]["feature_select_dir"]
    plots_dir     = config["paths"]["plots_dir"]
    top_n         = config["feature_selection"]["top_n"]
    method        = config["feature_selection"]["method"]
    project       = config.get("project", {})
    disease_label = project.get("disease_label", "SONFH")
    control_label = project.get("control_label", "control")
    dataset       = project.get("dataset", "")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log.info(f"📥 Loading preprocessed matrix: {input_path}")
    df = load_preprocessed(input_path)

    log.info(f"📊 Ranking probes by {method}, selecting top {top_n}")
    fc_ranking  = rank_by_fold_change(df, disease_label=disease_label, control_label=control_label)
    var_ranking = rank_by_variance(df)          # always needed for volcano plot
    ranking     = fc_ranking if method == "fc" else var_ranking

    top_probes  = ranking.head(top_n).index.tolist()
    selected_df = df[top_probes + ["class"]].copy()

    log.info(f"🧬 Loading probe annotations from SOFT file")
    gene_map = load_probe_annotation(soft_path)

    # --- CSVs ----------------------------------------------------------------

    top_csv = config["paths"]["top_features_csv"]
    selected_df.to_csv(top_csv)
    log.info(f"💾 Saved top features: {top_csv}  shape={selected_df.shape}")

    rankings_df = pd.DataFrame({
        "abs_log_fold_change": fc_ranking,
        "gene_symbol":         gene_map,
    })
    rankings_path = config["paths"]["gene_rankings"]
    rankings_df.to_csv(rankings_path)
    log.info(f"💾 Saved gene rankings: {rankings_path}")

    gene_summary_path = os.path.join(output_dir, "gene_level_summary.csv")
    build_gene_level_summary(selected_df, fc_ranking, gene_map, gene_summary_path,
                             disease_label=disease_label, control_label=control_label)
    log.info(f"💾 Saved gene-level summary: {gene_summary_path}")

    # --- EDA plots (same set as standalone feature_select.py) ----------------

    log.info("🖼️  Generating EDA plots...")

    eda_plots = [
        ("Volcano plot",          lambda: plot_volcano(
            fc_ranking, var_ranking, top_probes, gene_map,
            os.path.join(plots_dir, "volcano_plot.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
        ("Fold change bar chart", lambda: plot_fold_change_bar(
            fc_ranking,
            os.path.join(plots_dir, "fold_change_top20.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
        ("Box plots",             lambda: plot_boxplots(
            selected_df, gene_map,
            os.path.join(plots_dir, "boxplots_top6.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
        ("Sample correlation",    lambda: plot_sample_correlation(
            selected_df,
            os.path.join(plots_dir, "sample_correlation.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
        ("Heatmap",               lambda: plot_heatmap(
            selected_df,
            os.path.join(plots_dir, "heatmap_top20.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
        ("PCA",                   lambda: plot_pca(
            selected_df,
            os.path.join(plots_dir, "pca_plot.png"),
            disease_label=disease_label, control_label=control_label, dataset=dataset,
        )),
    ]

    for _label, fn in track(eda_plots, description="EDA plots", console=console):
        fn()

    log.info(f"✅ EDA plots saved to: {plots_dir}")

    return selected_df, fc_ranking, gene_map
