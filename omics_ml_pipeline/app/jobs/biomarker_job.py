"""
biomarker_job.py — Generate biomarker shortlist from RF feature importance + fold-change ranking.

Trains RandomForest across CV folds, records feature importance per fold,
computes selection frequency, merges with fold-change ranking and gene symbols,
outputs biomarker_shortlist.csv.
"""

import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.utils.logging_utils import get_logger

log = get_logger("biomarker_job")


def run(
    config: dict,
    selected_df: pd.DataFrame,
    fc_ranking: pd.Series,
    gene_map: pd.Series,
    run_id: str = "",
) -> pd.DataFrame:
    probe_cols = selected_df.columns[:-1].tolist()
    X = selected_df[probe_cols].values
    le = LabelEncoder()
    y = le.fit_transform(selected_df["class"].values)

    cv = StratifiedKFold(
        n_splits=config["training"]["cv_splits"],
        shuffle=True,
        random_state=config["training"]["random_state"],
    )

    log.info("Computing RF feature importance across CV folds...")
    importance_matrix = np.zeros((config["training"]["cv_splits"], len(probe_cols)))

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

    for fold_idx, (train_idx, _) in enumerate(cv.split(X, y)):
        rf.fit(X[train_idx], y[train_idx])
        importance_matrix[fold_idx] = rf.feature_importances_

    mean_importance = importance_matrix.mean(axis=0)
    selection_freq  = (importance_matrix > 0).mean(axis=0)

    shortlist = pd.DataFrame({
        "probe_id":        probe_cols,
        "gene_symbol":     [gene_map.get(p, "---") for p in probe_cols],
        "rf_importance":   mean_importance.round(4),
        "selection_freq":  selection_freq.round(3),
        "abs_fold_change": [fc_ranking.get(p, 0) for p in probe_cols],
    })

    # Combined score: average of normalised RF importance and normalised fold-change
    shortlist["rf_norm"] = shortlist["rf_importance"] / shortlist["rf_importance"].max()
    shortlist["fc_norm"] = shortlist["abs_fold_change"] / shortlist["abs_fold_change"].max()
    shortlist["combined_score"] = ((shortlist["rf_norm"] + shortlist["fc_norm"]) / 2).round(4)
    shortlist = shortlist.drop(columns=["rf_norm", "fc_norm"])
    shortlist = shortlist.sort_values("combined_score", ascending=False).reset_index(drop=True)

    top_n = config.get("biomarker", {}).get("top_n", len(shortlist))
    shortlist = shortlist.head(top_n)

    output_path = config["paths"]["biomarker_shortlist"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shortlist.to_csv(output_path, index=False)
    log.info(f"Saved biomarker shortlist: {output_path}  ({len(shortlist)} probes)")

    log.info("\nTop 10 candidates:")
    log.info(shortlist[["probe_id", "gene_symbol", "combined_score"]].head(10).to_string(index=False))

    with mlflow.start_run(run_name=f"{run_id}biomarker_shortlist"):
        mlflow.set_tag("stage", "biomarker")
        mlflow.log_metric("n_probes", len(shortlist))

    return shortlist
