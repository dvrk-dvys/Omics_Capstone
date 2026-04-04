"""
biomarker_job.py — Generate biomarker shortlist.

Multivariate mode
-----------------
  Trains RandomForest across CV folds, records feature importance per fold,
  computes selection frequency, merges with fold-change ranking and gene symbols,
  ranks by combined_score = (rf_norm + fc_norm) / 2, outputs biomarker_shortlist.csv.

Univariate baseline mode  (default when --mode univariate)
-----------------------------------------------------------
  Reads ann_probe_ranking.csv (already sorted by Median_TestAUC desc).
  Takes the top N probes directly from the ANN ranking.
  No RF loop. No second-stage reranking.
  Outputs biomarker_shortlist.csv with ANN-derived columns.

Univariate augmented mode  (--univariate-rerank flag)
------------------------------------------------------
  Reads ann_probe_ranking.csv, merges abs_fold_change from fc_ranking,
  computes univariate_score = 0.7*z(Median_TestAUC) + 0.3*z(abs_fold_change),
  sorts by univariate_score, takes top N.
  Outputs biomarker_shortlist.csv with univariate_score column added.
"""

import os
import numpy as np
import pandas as pd
import mlflow
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.utils.logging_utils import get_logger, console

log = get_logger("biomarker_job")


def _zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def run(
    config: dict,
    selected_df: pd.DataFrame,
    fc_ranking: pd.Series,
    gene_map: pd.Series,
    run_id: str = "",
) -> pd.DataFrame:

    mode             = config.get("_mode", "multivariate")
    univariate_rerank = config.get("_univariate_rerank", False)
    top_n            = config["biomarker"].get("top_n", 10)
    min_score        = config.get("biomarker", {}).get("min_score", 0.60)
    output_path      = config["paths"]["biomarker_shortlist"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ------------------------------------------------------------------
    # MULTIVARIATE PATH — RF importance + fold-change combined_score
    # ------------------------------------------------------------------
    if mode != "univariate":
        probe_cols = selected_df.columns[:-1].tolist()
        X = selected_df[probe_cols].values
        le = LabelEncoder()
        y = le.fit_transform(selected_df["class"].values)

        cv = StratifiedKFold(
            n_splits=config["training"]["cv_splits"],
            shuffle=True,
            random_state=config["training"]["random_state"],
        )

        log.info("🌲 Computing RF feature importance across CV folds...")
        importance_matrix = np.zeros((config["training"]["cv_splits"], len(probe_cols)))
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

        for fold_idx, (train_idx, _) in track(
            enumerate(cv.split(X, y)),
            description="RF importance folds",
            total=cv.n_splits,
            console=console,
        ):
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

        shortlist["rf_norm"] = shortlist["rf_importance"] / shortlist["rf_importance"].max()
        shortlist["fc_norm"] = shortlist["abs_fold_change"] / shortlist["abs_fold_change"].max()
        shortlist["combined_score"] = ((shortlist["rf_norm"] + shortlist["fc_norm"]) / 2).round(4)
        shortlist = shortlist.drop(columns=["rf_norm", "fc_norm"])
        shortlist = shortlist.sort_values("combined_score", ascending=False).reset_index(drop=True)
        shortlist = shortlist[shortlist["combined_score"] >= min_score]

        shortlist.to_csv(output_path, index=False)
        log.info(
            f"💾 Saved biomarker shortlist (multivariate): {output_path}"
            f"  ({len(shortlist)} probes, min_score≥{min_score})"
        )
        log.info(f"\n🎯 Top {top_n} candidates:")
        log.info(shortlist[["probe_id", "gene_symbol", "combined_score"]].head(top_n).to_string(index=False))

        with mlflow.start_run(run_name=f"{run_id}biomarker_shortlist"):
            mlflow.set_tag("stage", "biomarker")
            mlflow.set_tag("shortlist_mode", "multivariate")
            mlflow.log_metric("n_probes", len(shortlist))

        return shortlist

    # ------------------------------------------------------------------
    # UNIVARIATE PATH — ANN-derived ranking
    # ------------------------------------------------------------------
    ranking_path = config["paths"]["univariate_ann_ranking"]
    log.info(f"📥 Reading ANN probe ranking: {ranking_path}")
    ann_df = pd.read_csv(ranking_path)

    # Merge abs_fold_change from fc_ranking (needed for augmented mode and
    # for the column to be available in baseline output too)
    ann_df["abs_fold_change"] = ann_df["probe_id"].map(fc_ranking).fillna(0.0).round(4)

    # Ensure gene_symbol is present (ann_probe_ranking.csv should have it; fallback to gene_map)
    if "gene_symbol" not in ann_df.columns:
        ann_df["gene_symbol"] = ann_df["probe_id"].map(gene_map).fillna("---")

    n_select = config["feature_selection"]["top_n_feats"]

    if not univariate_rerank:
        # --------------------------------------------------------------
        # UNIVARIATE BASELINE — sort by Median_TestAUC, take top N as-is
        # --------------------------------------------------------------
        log.info(f"🧬 Univariate baseline: taking top {n_select} probes by Median_TestAUC")
        shortlist = (
            ann_df
            .sort_values("Median_TestAUC", ascending=False)
            .head(n_select)
            .reset_index(drop=True)
        )

        keep_cols = ["probe_id", "gene_symbol", "Median_TestAUC"]
        if "SD_TestAUC" in shortlist.columns:
            keep_cols.append("SD_TestAUC")
        keep_cols.append("abs_fold_change")
        shortlist = shortlist[[c for c in keep_cols if c in shortlist.columns]]

        shortlist.to_csv(output_path, index=False)
        log.info(
            f"💾 Saved biomarker shortlist (univariate_baseline): {output_path}"
            f"  ({len(shortlist)} probes)"
        )
        log.info(f"\n🎯 Top {top_n} candidates:")
        log.info(shortlist[["probe_id", "gene_symbol", "Median_TestAUC"]].head(top_n).to_string(index=False))

        with mlflow.start_run(run_name=f"{run_id}biomarker_shortlist"):
            mlflow.set_tag("stage", "biomarker")
            mlflow.set_tag("shortlist_mode", "univariate_baseline")
            mlflow.log_metric("n_probes", len(shortlist))

        return shortlist

    else:
        # --------------------------------------------------------------
        # UNIVARIATE AUGMENTED — rerank by univariate_score then take top N
        # univariate_score = 0.7 * z(Median_TestAUC) + 0.3 * z(abs_fold_change)
        # --------------------------------------------------------------
        log.info(
            f"🧬 Univariate augmented: reranking by "
            f"univariate_score = 0.7*z(Median_TestAUC) + 0.3*z(abs_fold_change), "
            f"taking top {n_select}"
        )
        ann_df["univariate_score"] = (
            0.7 * _zscore(ann_df["Median_TestAUC"]) +
            0.3 * _zscore(ann_df["abs_fold_change"])
        ).round(4)

        shortlist = (
            ann_df
            .sort_values("univariate_score", ascending=False)
            .head(n_select)
            .reset_index(drop=True)
        )

        keep_cols = ["probe_id", "gene_symbol", "Median_TestAUC"]
        if "SD_TestAUC" in shortlist.columns:
            keep_cols.append("SD_TestAUC")
        keep_cols += ["abs_fold_change", "univariate_score"]
        shortlist = shortlist[[c for c in keep_cols if c in shortlist.columns]]

        shortlist.to_csv(output_path, index=False)
        log.info(
            f"💾 Saved biomarker shortlist (univariate_augmented): {output_path}"
            f"  ({len(shortlist)} probes)"
        )
        log.info(f"\n🎯 Top {top_n} candidates:")
        log.info(
            shortlist[["probe_id", "gene_symbol", "univariate_score", "Median_TestAUC"]]
            .head(top_n).to_string(index=False)
        )

        with mlflow.start_run(run_name=f"{run_id}biomarker_shortlist"):
            mlflow.set_tag("stage", "biomarker")
            mlflow.set_tag("shortlist_mode", "univariate_augmented")
            mlflow.log_metric("n_probes", len(shortlist))

        return shortlist
