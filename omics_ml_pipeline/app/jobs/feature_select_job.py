"""
feature_select_job.py — Orchestrates feature selection with config-driven paths.

Imports primitive functions from feature_select.py directly rather than
calling select_top_probes(), which has a hardcoded SOFT_GZ path internally.
"""

import os
import pandas as pd

from app.utils.feature_select import (
    load_preprocessed,
    rank_by_fold_change,
    rank_by_variance,
    load_probe_annotation,
)
from app.utils.logging_utils import get_logger

log = get_logger("feature_select_job")


def run(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
        selected_df  : DataFrame — top N probes + class column
        fc_ranking   : Series   — full fold-change ranking (all probes)
        gene_map     : Series   — probe_id → gene_symbol
    """
    input_path  = config["paths"]["preprocessed_csv"]
    soft_path   = config["paths"]["soft_file"]
    output_dir  = config["paths"]["feature_select_dir"]
    top_n       = config["feature_selection"]["top_n"]
    method      = config["feature_selection"]["method"]

    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading preprocessed matrix: {input_path}")
    df = load_preprocessed(input_path)

    log.info(f"Ranking probes by {method}, selecting top {top_n}")
    fc_ranking = rank_by_fold_change(df)
    ranking    = fc_ranking if method == "fc" else rank_by_variance(df)

    top_probes  = ranking.head(top_n).index.tolist()
    selected_df = df[top_probes + ["class"]].copy()

    log.info(f"Loading probe annotations from SOFT file")
    gene_map = load_probe_annotation(soft_path)

    # Save top features CSV
    top_csv = config["paths"]["top_features_csv"]
    selected_df.to_csv(top_csv)
    log.info(f"Saved top features: {top_csv}  shape={selected_df.shape}")

    # Save full gene rankings
    rankings_df = pd.DataFrame({
        "abs_log_fold_change": fc_ranking,
        "gene_symbol": gene_map,
    })
    rankings_path = config["paths"]["gene_rankings"]
    rankings_df.to_csv(rankings_path)
    log.info(f"Saved gene rankings: {rankings_path}")

    return selected_df, fc_ranking, gene_map
