"""
feature_select_job.py — Orchestrates feature selection with config-driven paths.

Imports primitive functions from feature_select.py directly rather than
calling select_top_probes(), which has a hardcoded SOFT_GZ path internally.

Produces the following outputs:
  - gene_rankings.csv
  - gene_level_summary.csv
  - gene_level_rankings.csv
  - top100_genes.csv
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
    rank_by_hybrid_score,
    load_probe_annotation,
    build_gene_level_summary,
    build_gene_level_dedup,
    plot_volcano,
    plot_fold_change_bar,
    plot_boxplots,
    plot_sample_correlation,
    plot_heatmap,
    plot_pca,
    plot_composite_eda,
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
    top_n         = config["feature_selection"]["top_n_feats"]
    method        = config["feature_selection"]["method"]
    project       = config.get("project", {})
    disease_label = project.get("disease_label", "SONFH")
    control_label = project.get("control_label", "control")
    dataset       = f"{project.get('dataset', '')} — top {top_n} probes"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log.info(f"📥 Loading preprocessed matrix: {input_path}")
    df = load_preprocessed(input_path)

    log.info(f"📊 Ranking probes by multivariate score Z(|FC|) + Z(|t|), selecting top {top_n}")
    fc_ranking  = rank_by_fold_change(df, disease_label=disease_label, control_label=control_label)
    var_ranking = rank_by_variance(df)          # kept for volcano plot
    hybrid_df   = rank_by_hybrid_score(df, disease_label=disease_label, control_label=control_label)

    top_probes  = hybrid_df.head(top_n).index.tolist()
    selected_df = df[top_probes + ["class"]].copy()

    log.info(f"🧬 Loading probe annotations from SOFT file")
    gene_map = load_probe_annotation(soft_path)

    # --- CSVs ----------------------------------------------------------------

    # gene_rankings.csv — full probe ranking with all hybrid metrics
    ranked = hybrid_df.copy()
    ranked.index.name = "probe_id"
    ranked.insert(0, "probe_rank", range(1, len(ranked) + 1))
    ranked["gene_symbol"] = ranked.index.map(gene_map).fillna("---")
    _col_order = ["probe_rank", "gene_symbol", "multivariate_score", "abs_fold_change",
                  "log_fold_change", "t_stat", "p_value", "iqr", "variance",
                  "mean_sonfh", "mean_control", "probe_type"]
    rankings_path = config["paths"]["gene_rankings"]
    ranked[[c for c in _col_order if c in ranked.columns]].to_csv(rankings_path)
    log.info(f"💾 Saved gene rankings: {rankings_path}  ({len(ranked)} probes, all metrics)")

    gene_summary_path = os.path.join(output_dir, "gene_level_summary.csv")
    build_gene_level_summary(selected_df, fc_ranking, gene_map, gene_summary_path,
                             disease_label=disease_label, control_label=control_label)
    log.info(f"💾 Saved gene-level summary: {gene_summary_path}")

    # gene_level_rankings.csv + top100_genes.csv — gene-level deduped for interpretation
    gene_level_path = os.path.join(output_dir, "gene_level_rankings.csv")
    top_genes_path  = os.path.join(output_dir, "top100_genes.csv")
    build_gene_level_dedup(hybrid_df, gene_map, gene_level_path, top_genes_path, top_n=100)
    log.info(f"💾 Saved gene-level deduped rankings: {gene_level_path}")

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
            gene_map=gene_map,
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

    # --- EDA composite multi-panel figure ------------------------------------
    composite_path = os.path.join(plots_dir, "fig_1_eda_composite.png")
    plot_composite_eda(plots_dir, composite_path, dataset=f"{dataset} (Python pipeline)")
    log.info(f"💾 EDA composite figure saved: {composite_path}")

    return selected_df, fc_ranking, gene_map
