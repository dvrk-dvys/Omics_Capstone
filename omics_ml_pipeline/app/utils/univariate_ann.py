"""
univariate_ann.py — Professor-faithful univariate ANN wrapper.

Faithful Python port of:
  r_base_scripts/simple ANN wrapper and filter.R

TWO USAGE MODES
---------------
1. Standalone utility (Weka prep path):

     python -m app.utils.univariate_ann
     python -m app.utils.univariate_ann --input path/to/preprocessed.csv --outdir path/to/out

   Reads the preprocessed matrix, runs the univariate filter + wrapper, and writes:

     PRIMARY Weka output (course-faithful / report-faithful):
       top100_features_univariate_ann.csv / .arff  <- default Weka input for the course workflow

     Optional exploratory output (not the default Weka/report path):
       top500_features_univariate_ann.csv / .arff  <- only use if explicitly choosing to explore

     Pipeline performance CSVs:
       filter_univariate_auc.csv
       wrapper_performance.csv
       wrapper_summary_per_gene.csv
       wrapper_predictions.csv

     Downstream LLM input:
       univariate_biomarker_shortlist.csv

   Default input  : data/femoral_head_necrosis/parsed/preprocessed_matrix.csv
                    (output of preprocess.py in the Weka standalone workflow)
   Default outdir : data/femoral_head_necrosis/univariate_ann/

2. Reusable module (app pipeline):

     from app.utils.univariate_ann import run_univariate_pipeline
     Called by app/jobs/univariate_ann_job.py with config-driven paths.

METHODOLOGY  (mirrors r_base_scripts/simple ANN wrapper and filter.R)
----------------------------------------------------------------------
  - One probe at a time — unit of modelling = one probe / one ANN
  - Optional univariate AUC filter on full dataset (no split) — screens weak probes
  - N_MCCV x stratified 70/30 Monte Carlo CV splits per probe
  - Tiny ANN per probe per split:
        Linear(1->8) -> ReLU -> Dropout(0.2) -> Linear(8->4) -> ReLU -> Linear(4->1) -> Sigmoid
  - Scale fit on TRAIN only; same transform applied to TEST
  - Validation split = last 20% of training rows (exact Keras validation_split=0.2 behaviour)
  - Early stopping: monitor val_loss, patience=8, restore best weights
  - Rank probes by median held-out TestAUC (tiebreak: median TestAcc)
"""

import argparse
import copy
import gzip
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from app.models.univariate_models import get_univariate_models


# ---------------------------------------------------------------------------
# MODULE-LEVEL PATH CONSTANTS
# Used only by the standalone __main__ entry point.
# Job mode uses config-driven paths from pipeline.yaml.
# ---------------------------------------------------------------------------
_APP_DIR      = pathlib.Path(__file__).resolve().parent.parent   # omics_ml_pipeline/app/
_PROJECT_ROOT = _APP_DIR.parent.parent                            # Omics_Capstone/

# Weka standalone defaults — mirror preprocess.py / feature_select.py path conventions
_WEKA_INPUT_CSV  = str(
    _PROJECT_ROOT / "data" / "femoral_head_necrosis" / "parsed" / "preprocessed_matrix.csv"
)
_WEKA_SOFT_GZ    = str(_APP_DIR / "data" / "input" / "GSE123568_family.soft.gz")
_WEKA_OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "feature_selection" / "univariate_ann")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_preprocessed(path: str) -> pd.DataFrame:
    """Load preprocessed probe matrix CSV (samples x probes + class column)."""
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded: {df.shape[0]} samples x {df.shape[1] - 1} probes (+class)")
    return df


def load_probe_annotation(soft_gz_path: str) -> pd.Series:
    """
    Read the platform annotation table from the SOFT .gz file.
    Extracts rows between !platform_table_begin and !platform_table_end.
    Returns a Series mapping probe_id -> gene_symbol.

    Copied verbatim from app/utils/feature_select.py to keep this module
    self-contained (avoids importing the heavy feature_select module which
    triggers matplotlib/scipy/sklearn side effects at import time).
    """
    if not os.path.exists(soft_gz_path):
        print(f"  Note: SOFT file not found at {soft_gz_path} — gene names unavailable")
        return pd.Series(dtype=str)

    rows = []
    header = None
    in_table = False

    with gzip.open(soft_gz_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            elif line == "!platform_table_end":
                break
            elif in_table:
                parts = line.split("\t")
                if header is None:
                    header = parts
                else:
                    rows.append(parts)

    if not header or not rows:
        print("  Note: could not parse platform table from SOFT file")
        return pd.Series(dtype=str)

    ann = pd.DataFrame(rows, columns=header).set_index("ID")
    col = "Gene Symbol" if "Gene Symbol" in ann.columns else ann.columns[0]
    return ann[col].fillna("---")


# ---------------------------------------------------------------------------
# METRIC HELPERS — direct ports of R safe_auc() and safe_accuracy()
# ---------------------------------------------------------------------------

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Compute AUC. Returns None when fewer than 2 unique classes in y_true.
    Mirrors R safe_auc().
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    Threshold predictions and return accuracy. Returns None if y_true is empty.
    Mirrors R safe_accuracy().
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return None
    y_hat = (y_prob >= threshold).astype(float)
    return float(np.mean(y_hat == y_true))


# ---------------------------------------------------------------------------
# SCALING — direct port of R scale_train_apply()
# ---------------------------------------------------------------------------

def scale_train_apply(x_train: np.ndarray, x_test: np.ndarray):
    """
    Fit z-score normalisation on x_train; apply the same transform to x_test.
    If std == 0, substitute std = 1 (mirrors R sdv fallback).
    Returns (scaled_train, scaled_test, mu, sd).
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test  = np.asarray(x_test,  dtype=float)
    mu = float(np.nanmean(x_train))
    sd = float(np.nanstd(x_train, ddof=1)) if len(x_train) > 1 else 0.0
    if np.isnan(sd) or sd == 0.0:
        sd = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# STRATIFIED SPLIT — direct port of R mc_split_stratified()
# ---------------------------------------------------------------------------

def mc_split_stratified(
    y01: np.ndarray,
    train_frac: float = 0.70,
    rng: np.random.Generator = None,
):
    """
    Stratified random 70/30 split.
    Samples floor(n_pos * train_frac) positives and floor(n_neg * train_frac) negatives
    independently. Guarantees at least 1 of each class in train.
    test = complement of train.
    Mirrors R mc_split_stratified().
    """
    if rng is None:
        rng = np.random.default_rng(123)
    y01 = np.asarray(y01, dtype=int)
    idx_pos = np.where(y01 == 1)[0]
    idx_neg = np.where(y01 == 0)[0]

    n_pos_train = max(1, int(np.floor(len(idx_pos) * train_frac)))
    n_neg_train = max(1, int(np.floor(len(idx_neg) * train_frac)))

    train_pos = rng.choice(idx_pos, size=n_pos_train, replace=False)
    train_neg = rng.choice(idx_neg, size=n_neg_train, replace=False)

    train_idx = np.concatenate([train_pos, train_neg])
    test_idx  = np.setdiff1d(np.arange(len(y01)), train_idx)
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# ANN MODEL — direct port of R build_single_gene_model()
#
# Architecture:
#   Linear(1->8) -> ReLU -> Dropout(0.2) -> Linear(8->4) -> ReLU -> Linear(4->1) -> Sigmoid
# ---------------------------------------------------------------------------

class SingleGeneNet(nn.Module):
    """
    Tiny ANN for univariate (1-feature) binary classification.
    Faithful PyTorch port of R build_single_gene_model() (Keras 3).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_single_gene_model() -> SingleGeneNet:
    """
    Return a freshly initialised SingleGeneNet.
    Mirrors R: keras::k_clear_session() + build_single_gene_model().
    """
    return SingleGeneNet()


# ---------------------------------------------------------------------------
# TRAINING — mirrors R model |> fit(..., validation_split=0.2, callbacks=[early_stop])
# ---------------------------------------------------------------------------

def train_single_gene_model(
    model:      SingleGeneNet,
    x_train:    np.ndarray,
    y_train:    np.ndarray,
    epochs:     int   = 60,
    batch_size: int   = 16,
    patience:   int   = 8,
    lr:         float = 0.001,
    val_frac:   float = 0.20,
) -> None:
    """
    Train model in-place.

    Validation split:
      Takes the LAST floor(n_train * val_frac) rows of x_train as validation.
      Does NOT shuffle before splitting.
      Mirrors Keras validation_split=0.2 exact behaviour.

    Early stopping:
      Monitors val_loss, patience=8, restores best weights at end.
      Mirrors Keras callback_early_stopping(restore_best_weights=TRUE).

    Args:
        x_train : (n_train, 1) float array — full training set incl. the val portion.
        y_train : (n_train,)   float array — binary 0/1 labels.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n     = len(x_train)
    val_n = int(np.floor(n * val_frac))   # floor matches Keras behaviour

    # Split: LAST val_n rows -> validation; remainder -> mini-batch training
    if val_n > 0:
        x_tr  = x_train[:-val_n]
        y_tr  = y_train[:-val_n]
        x_val = x_train[-val_n:]
        y_val = y_train[-val_n:]
    else:
        x_tr  = x_train
        y_tr  = y_train
        x_val = None
        y_val = None

    X_tr = torch.FloatTensor(x_tr)
    Y_tr = torch.FloatTensor(y_tr)

    X_val = torch.FloatTensor(x_val) if x_val is not None else None
    Y_val = torch.FloatTensor(y_val) if y_val is not None else None

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for _epoch in range(epochs):
        # --- mini-batch training pass ---
        model.train()
        n_tr = len(X_tr)
        for i in range(0, n_tr, batch_size):
            batch_x = X_tr[i : i + batch_size]
            batch_y = Y_tr[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(batch_x).squeeze(-1)   # (batch,)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # --- validation + early stopping ---
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze(-1)
                val_loss = criterion(val_pred, Y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(model.state_dict())
                no_improve    = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

    # Restore best weights — mirrors restore_best_weights=TRUE
    if best_state is not None:
        model.load_state_dict(best_state)


# ---------------------------------------------------------------------------
# MODEL DISPATCH — fit and predict for any registered univariate model type
# ---------------------------------------------------------------------------

def fit_univariate_model(
    model_type: str,
    x_train:    np.ndarray,
    y_train:    np.ndarray,
    cfg:        dict,
):
    """
    Fit one selected univariate model and return the fitted model.

    professor_ann  — uses the existing faithful PyTorch implementation.
    logistic       — fresh LogisticRegression from univariate_models registry.
    svm_rbf        — fresh SVC(rbf) from univariate_models registry.

    Args:
        x_train : (n_train, 1) float32 array — already z-score scaled upstream.
        y_train : (n_train,)   float32 array — binary 0/1 labels.
        cfg     : parameter dict; ANN uses epochs/batch_size/patience/learning_rate/val_frac.
                  sklearn models use defaults from the registry (class_weight='balanced').
    """
    if model_type == "professor_ann":
        model = build_single_gene_model()
        train_single_gene_model(
            model, x_train, y_train,
            epochs=cfg.get("epochs", 60),
            batch_size=cfg.get("batch_size", 16),
            patience=cfg.get("patience", 8),
            lr=cfg.get("learning_rate", 0.001),
            val_frac=cfg.get("val_frac", 0.20),
        )
        return model

    registry = get_univariate_models(random_state=cfg.get("seed", 42))
    if model_type not in registry:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from: professor_ann, {', '.join(registry.keys())}"
        )
    model = registry[model_type]
    model.fit(x_train, y_train.astype(int))
    return model


def predict_univariate_scores(
    model_type: str,
    model,
    x:          np.ndarray,
) -> np.ndarray:
    """
    Return probability-like scores in [0, 1] for AUC computation.

    professor_ann  — sigmoid output from the trained PyTorch net.
    logistic/svm_rbf — predict_proba[:, 1] (positive-class probability).
    Fallback        — min-max scaled decision_function if predict_proba unavailable.
    """
    if model_type == "professor_ann":
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(x)).squeeze(-1).numpy()

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        smin, smax = scores.min(), scores.max()
        if smax - smin == 0:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - smin) / (smax - smin)

    raise ValueError(
        f"Model type '{model_type}' supports neither predict_proba nor decision_function."
    )


# ---------------------------------------------------------------------------
# FILTER STEP — direct port of R FILTER block (lines 144-166)
# ---------------------------------------------------------------------------

def run_filter(
    df_features: pd.DataFrame,
    y01:         np.ndarray,
    probe_cols:  list,
    top_n:       int = 2000,
):
    """
    Compute univariate AUC for each probe on the FULL dataset (no train/test split).
    Screening step only — AUC computed from z-scored + min-max-scaled probe values.

    Per probe:
      1. z-score normalise using mean / sd of non-NA values
      2. min-max scale to [0, 1]
      3. AUC(y_true, score01)

    Probes with fewer than 10 non-NA values are kept in the output table with
    FilterAUC=None but are NOT included in the selected probe list.
    Only probes with a valid (non-null) FilterAUC enter the wrapper.

    Mirrors R filter step (lines 144-166 of simple ANN wrapper and filter.R).

    Returns:
        filter_df     : DataFrame[Gene, FilterAUC], all probes, sorted desc by FilterAUC.
        selected_cols : top_n probe IDs with non-null FilterAUC (or all if fewer exist).
    """
    records = []
    for g in probe_cols:
        x  = df_features[g].values.astype(float)
        ok = ~np.isnan(x)
        if ok.sum() < 10:
            # mirrors R: if (sum(ok) < 10) next
            records.append({"Gene": g, "FilterAUC": None})
            continue

        x_ok  = x[ok]
        mu    = x_ok.mean()
        sd    = x_ok.std()
        xz    = (x_ok - mu) / (sd + 1e-8)
        xmin  = xz.min()
        xmax  = xz.max()
        score = (xz - xmin) / (xmax - xmin + 1e-8)

        records.append({"Gene": g, "FilterAUC": safe_auc(y01[ok], score)})

    filter_df = (
        pd.DataFrame(records)
        .sort_values("FilterAUC", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    # Only probes with non-null FilterAUC enter the wrapper
    selected_cols = (
        filter_df.loc[filter_df["FilterAUC"].notna(), "Gene"]
        .head(top_n)
        .tolist()
    )
    return filter_df, selected_cols


# ---------------------------------------------------------------------------
# GENE SUMMARY — direct port of R summary_per_gene aggregation
# ---------------------------------------------------------------------------

def build_gene_summary(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-iteration metrics to per-probe summary.

    Mirrors R:
      summarise(N_runs, Median_TestAUC, Mean_TestAUC, SD_TestAUC, Median_TestAcc)
      arrange(desc(Median_TestAUC), desc(Median_TestAcc))
    """
    summary = (
        perf_df.groupby("Gene", sort=False)
        .agg(
            N_runs        =("TestAUC", lambda x: int(x.notna().sum())),
            Median_TestAUC=("TestAUC", "median"),
            Mean_TestAUC  =("TestAUC", "mean"),
            SD_TestAUC    =("TestAUC", "std"),
            Median_TestAcc=("TestAcc", "median"),
        )
        .reset_index()
        .sort_values(
            ["Median_TestAUC", "Median_TestAcc"],
            ascending=[False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    return summary


# ---------------------------------------------------------------------------
# WRAPPER — direct port of R WRAPPER block (lines 172-260)
# ---------------------------------------------------------------------------

def run_wrapper(
    df_features:  pd.DataFrame,
    y01:          np.ndarray,
    probe_cols:   list,
    model_type:   str   = "professor_ann",
    model_cfg:    dict  = None,
    n_mccv:       int   = 5,
    train_frac:   float = 0.70,
    seed:         int   = 123,
    epochs:       int   = 60,
    batch_size:   int   = 16,
    patience:     int   = 8,
    lr:           float = 0.001,
    val_frac:     float = 0.20,
    min_train:    int   = 10,
    min_test:     int   = 5,
    log_fn=None,
):
    """
    Single-gene ANN wrapper with Monte Carlo CV.

    For each probe in probe_cols, for each of n_mccv iterations:
      1. Stratified 70/30 random split
      2. Drop NAs per split
      3. Skip if: train < min_train, test < min_test,
                  or single class in train or test after NA removal
      4. Scale: fit z-score on train, apply to test
      5. Build + train SingleGeneNet
      6. Predict on full train + test sets
      7. Record TrainAUC, TestAUC, TrainAcc, TestAcc

    Single-threaded by design — mirrors R sequential loop.
    Single RNG with fixed seed — mirrors R set.seed(SEED).

    Returns:
        perf_df    : per-probe per-iteration metrics DataFrame
        summary_df : per-probe aggregated summary, sorted by Median_TestAUC desc
        pred_df    : per-probe per-split sample-level predictions DataFrame
    """
    if log_fn is None:
        log_fn = print

    # Build model config dict used by fit_univariate_model dispatch.
    # For professor_ann: ANN hyperparams from individual params (backward compat).
    # For logistic/svm_rbf: model_cfg or empty dict (defaults live in the registry).
    _model_cfg = model_cfg if model_cfg is not None else {
        "epochs": epochs, "batch_size": batch_size,
        "patience": patience, "learning_rate": lr, "val_frac": val_frac,
        "seed": seed,
    }

    # Single RNG — mirrors R set.seed(SEED) at top of script
    rng = np.random.default_rng(seed)

    n_total = len(probe_cols) * n_mccv
    log_fn(
        f"Running wrapper ANN: {len(probe_cols)} probes x {n_mccv} MCCV "
        f"= {n_total} model fits"
    )

    perf_rows = []
    pred_rows = []

    for probe_num, g in enumerate(probe_cols, start=1):
        if probe_num % 100 == 0 or probe_num == len(probe_cols):
            log_fn(f"  Probe {probe_num}/{len(probe_cols)}: {g}")

        x_full = df_features[g].values.astype(float)

        for iteration in range(1, n_mccv + 1):
            train_idx, test_idx = mc_split_stratified(
                y01, train_frac=train_frac, rng=rng
            )

            # Drop NAs per split — mirrors R train_ok / test_ok
            train_ok = train_idx[~np.isnan(x_full[train_idx])]
            test_ok  = test_idx[ ~np.isnan(x_full[test_idx])]

            # --- skip conditions (mirrors R lines 194-210) ---
            skip = ""
            if len(train_ok) < min_train:
                skip = "Skipped: insufficient data or single-class after NA removal"
            elif len(test_ok) < min_test:
                skip = "Skipped: insufficient data or single-class after NA removal"
            elif len(np.unique(y01[train_ok])) < 2:
                skip = "Skipped: insufficient data or single-class after NA removal"
            elif len(np.unique(y01[test_ok])) < 2:
                skip = "Skipped: insufficient data or single-class after NA removal"

            if skip:
                perf_rows.append({
                    "Gene":      g,
                    "Iteration": iteration,
                    "N_train":   len(train_ok),
                    "N_test":    len(test_ok),
                    "TrainAUC":  None,
                    "TestAUC":   None,
                    "TrainAcc":  None,
                    "TestAcc":   None,
                    "ScaleMean": None,
                    "ScaleSD":   None,
                    "Notes":     skip,
                })
                continue

            # --- scale (mirrors R scale_train_apply) ---
            x_tr_sc, x_te_sc, mu, sd = scale_train_apply(
                x_full[train_ok], x_full[test_ok]
            )

            # Reshape to (n, 1) for ANN input_shape = c(1)
            x_train = x_tr_sc.reshape(-1, 1).astype(np.float32)
            y_train = y01[train_ok].astype(np.float32)
            x_test  = x_te_sc.reshape(-1, 1).astype(np.float32)
            y_test  = y01[test_ok].astype(np.float32)

            # --- fit + predict (model_type-dispatched; professor_ann path is professor-faithful) ---
            _fitted = fit_univariate_model(model_type, x_train, y_train, _model_cfg)
            p_train = np.atleast_1d(predict_univariate_scores(model_type, _fitted, x_train))
            p_test  = np.atleast_1d(predict_univariate_scores(model_type, _fitted, x_test))

            perf_rows.append({
                "Gene":      g,
                "Iteration": iteration,
                "N_train":   len(train_ok),
                "N_test":    len(test_ok),
                "TrainAUC":  safe_auc(y_train, p_train),
                "TestAUC":   safe_auc(y_test,  p_test),
                "TrainAcc":  safe_accuracy(y_train, p_train),
                "TestAcc":   safe_accuracy(y_test,  p_test),
                "ScaleMean": float(mu),
                "ScaleSD":   float(sd),
                "Notes":     "",
            })

            # predictions table — mirrors R pred_rows
            all_indices = np.concatenate([train_ok, test_ok])
            all_splits  = (["Train"] * len(train_ok)) + (["Test"] * len(test_ok))
            all_y_true  = np.concatenate([y_train, y_test])
            all_y_prob  = np.concatenate([p_train, p_test])

            for idx, split_label, yt, yp in zip(
                all_indices, all_splits, all_y_true, all_y_prob
            ):
                pred_rows.append({
                    "Gene":        g,
                    "Iteration":   iteration,
                    "SampleIndex": int(idx),
                    "Split":       split_label,
                    "y_true":      float(yt),
                    "y_prob":      float(yp),
                })

    perf_df    = pd.DataFrame(perf_rows)
    pred_df    = pd.DataFrame(pred_rows)
    summary_df = build_gene_summary(perf_df)

    log_fn(
        f"Wrapper complete. Valid runs: "
        f"{int(perf_df['TestAUC'].notna().sum())} / {len(perf_df)}"
    )
    return perf_df, summary_df, pred_df


# ---------------------------------------------------------------------------
# ARFF EXPORT
# Copied verbatim from app/utils/feature_select.py write_arff() to keep
# this module self-contained (avoids importing the heavy feature_select module).
# Format is identical — same Weka ARFF convention used throughout this repo.
# ---------------------------------------------------------------------------

def write_arff(df: pd.DataFrame, relation_name: str, path: str) -> None:
    """
    Write a DataFrame to Weka ARFF format.

    Convention (identical to feature_select.py):
      @RELATION  name
      @ATTRIBUTE probe_id NUMERIC       <- one per selected probe (all non-class columns)
      ...
      @ATTRIBUTE class {label_a,label_b} <- nominal, last attribute
      @DATA
      val,val,...,class_label           <- one row per sample
    """
    gene_cols    = df.columns[:-1]
    class_values = sorted(df["class"].unique().tolist())

    with open(path, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for gene in gene_cols:
            safe_name = str(gene).replace("-", "_").replace(".", "_")
            f.write(f"@ATTRIBUTE {safe_name} NUMERIC\n")

        class_str = ",".join(class_values)
        f.write(f"@ATTRIBUTE class {{{class_str}}}\n")
        f.write("\n@DATA\n")

        for _, row in df.iterrows():
            gene_vals = ",".join(f"{v:.6f}" for v in row[gene_cols])
            class_val = row["class"]
            f.write(f"{gene_vals},{class_val}\n")

    print(f"ARFF written: {path}")
    print(f"  {len(gene_cols)} features + class {{{class_str}}}")
    print(f"  {len(df)} instances")


# ---------------------------------------------------------------------------
# WEKA OUTPUT WRITER
# ---------------------------------------------------------------------------

def write_weka_outputs(
    df:            pd.DataFrame,
    summary_df:    pd.DataFrame,
    output_dir:    str,
    top_ns:        list = None,
    relation_name: str = "univariate_ann",
) -> None:
    """
    Write Weka-ready CSV + ARFF for each value of top_n.

    PRIMARY output  : top100_features_univariate_ann.csv / .arff
                      Default Weka input for the course-faithful / report-faithful workflow.

    OPTIONAL output : top500_features_univariate_ann.csv / .arff
                      Exploratory only — not the default Weka/report path.
                      Only use top500 if explicitly choosing to explore a larger probe set.

    Probe selection:
      Only probes with N_runs > 0 AND non-null Median_TestAUC are eligible.
      summary_df is already sorted by Median_TestAUC desc, so head(n) gives
      the top_n best-performing probes.
      If fewer than top_n eligible probes exist, writes however many are available.

    Output files (per top_n in top_ns):
      top{n}_features_univariate_ann.csv
      top{n}_features_univariate_ann.arff
    """
    if top_ns is None:
        top_ns = [100, 500]

    os.makedirs(output_dir, exist_ok=True)

    # Only probes with valid wrapper results are eligible for Weka export
    valid_mask       = (summary_df["N_runs"] > 0) & (summary_df["Median_TestAUC"].notna())
    available_probes = summary_df.loc[valid_mask, "Gene"].tolist()

    if not available_probes:
        print("  Skipping Weka export: no probes with valid wrapper results.")
        return

    for top_n in top_ns:
        label = "PRIMARY" if top_n == 100 else "optional exploratory"
        selected = available_probes[:top_n]

        # Guard: keep only probes that exist as columns in df
        selected = [p for p in selected if p in df.columns]
        if not selected:
            print(f"  Skipping top{top_n} ({label}): no matching probe columns in data frame.")
            continue

        weka_df = df[selected + ["class"]].copy()

        csv_path  = os.path.join(output_dir, f"top{top_n}_features_univariate_ann.csv")
        arff_path = os.path.join(output_dir, f"top{top_n}_features_univariate_ann.arff")

        weka_df.to_csv(csv_path)
        print(
            f"Weka CSV ({label}): {csv_path}  "
            f"({len(selected)} probes x {len(weka_df)} samples)"
        )
        write_arff(weka_df, relation_name=relation_name, path=arff_path)


# ---------------------------------------------------------------------------
# FULL PIPELINE ORCHESTRATION — shared by standalone and job modes
# ---------------------------------------------------------------------------

def run_univariate_pipeline(
    preprocessed_csv:  str,
    soft_gz_path:      str,
    output_dir:        str,
    disease_label:     str   = "SONFH",
    control_label:     str   = "control",
    model_type:        str   = "professor_ann",
    use_filter:        bool  = True,
    filter_top_n:      int   = 2000,
    n_mccv:            int   = 5,
    train_frac:        float = 0.70,
    seed:              int   = 123,
    epochs:            int   = 60,
    batch_size:        int   = 16,
    patience:          int   = 8,
    lr:                float = 0.001,
    val_frac:          float = 0.20,
    min_train:         int   = 10,
    min_test:          int   = 5,
    max_na_fraction:   float = 0.20,
    weka_top_ns:       list  = None,
    relation_name:     str   = "univariate_ann",
    log_fn=None,
):
    """
    Full univariate ANN pipeline from preprocessed CSV to all outputs.

    Called by:
      app/jobs/univariate_ann_job.py  (config-driven paths, app pipeline)
      __main__ block below            (argparse-driven paths, standalone Weka mode)

    Steps:
      1. Load preprocessed CSV + gene annotations
      2. QC: drop probes with > max_na_fraction NA or zero variance
      3. Optional filter: top-N by univariate AUC on full dataset
      4. Wrapper: per-probe ANN MCCV
      5. Save wrapper CSVs (performance, summary, predictions, filter AUC)
      6. Save univariate_biomarker_shortlist.csv  (LLM-compatible)
      7. Save Weka-ready CSV + ARFF:
           top100 (PRIMARY — default Weka/report path)
           top500 (optional exploratory — only use if explicitly chosen)

    Returns:
        perf_df    : per-probe per-iteration performance DataFrame
        summary_df : per-probe aggregated summary, sorted by Median_TestAUC
        pred_df    : per-probe per-split predictions DataFrame
        gene_map   : Series probe_id -> gene_symbol
    """
    if log_fn is None:
        log_fn = print
    if weka_top_ns is None:
        weka_top_ns = [100, 500]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    log_fn(f"Loading preprocessed matrix: {preprocessed_csv}")
    df = load_preprocessed(preprocessed_csv)

    log_fn(f"Loading probe annotations: {soft_gz_path}")
    gene_map = load_probe_annotation(soft_gz_path)

    # 2. Extract y01 and probe columns
    y_raw       = df["class"].values
    y01         = (y_raw == disease_label).astype(int)
    probe_cols  = [c for c in df.columns if c != "class"]
    df_features = df[probe_cols]

    n_pos = int((y01 == 1).sum())
    n_neg = int((y01 == 0).sum())
    log_fn(f"Class distribution: {disease_label}={n_pos}  {control_label}={n_neg}")

    # 3. QC: NA fraction + zero variance (mirrors R na_frac / var0 checks)
    na_fracs  = df_features.isna().mean()
    zero_var  = df_features.std(ddof=1) == 0
    keep_mask = (na_fracs <= max_na_fraction) & (~zero_var)
    probe_cols_qc = [c for c in probe_cols if keep_mask[c]]
    log_fn(
        f"QC: {len(probe_cols)} probes -> {len(probe_cols_qc)} after "
        f"NA-fraction (<= {max_na_fraction}) + zero-variance filter"
    )

    # 4. Filter step (optional univariate AUC screening)
    filter_path = os.path.join(output_dir, "filter_univariate_auc.csv")

    if use_filter:
        log_fn(
            f"Running univariate AUC filter — keeping top {filter_top_n} "
            f"of {len(probe_cols_qc)} probes..."
        )
        filter_df, probe_cols_filtered = run_filter(
            df_features, y01, probe_cols_qc, top_n=filter_top_n
        )
        log_fn(
            f"  Filter: {len(probe_cols_qc)} -> {len(probe_cols_filtered)} probes selected"
        )
    else:
        probe_cols_filtered = probe_cols_qc
        filter_df = pd.DataFrame({"Gene": probe_cols_qc, "FilterAUC": None})

    filter_df.to_csv(filter_path, index=False)
    log_fn(f"Saved: {filter_path}")

    # 5. Wrapper
    perf_df, summary_df, pred_df = run_wrapper(
        df_features, y01, probe_cols_filtered,
        model_type=model_type,
        n_mccv=n_mccv, train_frac=train_frac, seed=seed,
        epochs=epochs, batch_size=batch_size,
        patience=patience, lr=lr, val_frac=val_frac,
        min_train=min_train, min_test=min_test,
        log_fn=log_fn,
    )

    # 6. Save wrapper CSVs
    perf_path    = os.path.join(output_dir, "wrapper_performance.csv")
    summary_path = os.path.join(output_dir, "wrapper_summary_per_gene.csv")
    pred_path    = os.path.join(output_dir, "wrapper_predictions.csv")

    perf_df.to_csv(perf_path,       index=False)
    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(pred_path,       index=False)

    log_fn(f"Saved: {perf_path}")
    log_fn(f"Saved: {summary_path}")
    log_fn(f"Saved: {pred_path}")

    # 7. Univariate biomarker shortlist (LLM-compatible)
    shortlist = summary_df[
        ["Gene", "Median_TestAUC", "Mean_TestAUC", "SD_TestAUC", "Median_TestAcc"]
    ].copy()
    shortlist = shortlist.rename(columns={"Gene": "probe_id"})
    shortlist["gene_symbol"]    = shortlist["probe_id"].map(gene_map).fillna("---")
    shortlist["combined_score"] = shortlist["Median_TestAUC"]   # [0,1], directly interpretable
    shortlist = shortlist.sort_values("combined_score", ascending=False).reset_index(drop=True)

    shortlist_path = os.path.join(output_dir, "ann_probe_ranking.csv")
    shortlist.to_csv(shortlist_path, index=False)
    log_fn(f"Saved: {shortlist_path}")

    # 8. Weka-ready CSV + ARFF
    # top100 = PRIMARY (course-faithful / report-faithful)
    # top500 = optional exploratory (not the default Weka/report path)
    log_fn("Writing Weka-ready outputs (top100 = PRIMARY, top500 = optional exploratory)...")
    write_weka_outputs(
        df, summary_df, output_dir,
        top_ns=weka_top_ns,
        relation_name=relation_name,
    )

    # Preview
    top10 = shortlist[["probe_id", "gene_symbol", "combined_score"]].head(10)
    log_fn("\nTop 10 probes by Median_TestAUC:")
    log_fn(top10.to_string(index=False))

    # 9. Per-probe AUC distribution figure
    _plot_probe_auc_distribution(
        summary_df, gene_map, output_dir, model_type,
        disease_label=disease_label, control_label=control_label,
        top_n_selected=weka_top_ns[0] if weka_top_ns else None,
    )

    return perf_df, summary_df, pred_df, gene_map


# ---------------------------------------------------------------------------
# STANDALONE CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Univariate ANN wrapper — professor-faithful single-probe ANN MCCV. "
            "Step 3 of the Weka prep workflow: "
            "parse_series_matrix.py -> preprocess.py -> univariate_ann.py -> Weka GUI. "
            "Produces Weka-ready CSV/ARFF + LLM shortlist. "
            "PRIMARY Weka output: top100_features_univariate_ann.arff. "
            "top500 is generated as optional exploratory output only."
        )
    )
    p.add_argument(
        "--input",  type=str, default=_WEKA_INPUT_CSV,
        help=(
            "Path to preprocessed_matrix.csv  "
            f"(default: {_WEKA_INPUT_CSV})"
        ),
    )
    p.add_argument(
        "--soft",   type=str, default=_WEKA_SOFT_GZ,
        help="Path to GSE123568_family.soft.gz for probe->gene_symbol annotations",
    )
    p.add_argument(
        "--outdir", type=str, default=_WEKA_OUTPUT_DIR,
        help=(
            "Output directory for all generated files  "
            f"(default: {_WEKA_OUTPUT_DIR})"
        ),
    )
    p.add_argument(
        "--model-type", type=str, default="all",
        choices=["professor_ann", "logistic", "svm_rbf", "all"],
        help="Model type  (default: all). Runs all three models and saves comparison outputs."
    )
    p.add_argument("--disease-label",   type=str,   default="SONFH",   help="Positive class label  (default: SONFH)")
    p.add_argument("--control-label",   type=str,   default="control",  help="Negative class label  (default: control)")
    p.add_argument("--no-filter",       action="store_true",             help="Disable univariate AUC pre-filter step")
    p.add_argument("--filter-top-n",    type=int,   default=2000,        help="Max probes after filter  (default 2000)")
    p.add_argument("--n-mccv",          type=int,   default=5,           help="MCCV iterations per probe  (default 5)")
    p.add_argument("--train-frac",      type=float, default=0.70,        help="Train fraction  (default 0.70)")
    p.add_argument("--seed",            type=int,   default=123,         help="Random seed  (default 123)")
    p.add_argument("--epochs",          type=int,   default=60,          help="Max training epochs per ANN  (default 60)")
    p.add_argument("--batch-size",      type=int,   default=16,          help="Mini-batch size  (default 16)")
    p.add_argument("--patience",        type=int,   default=8,           help="Early stopping patience  (default 8)")
    p.add_argument("--lr",              type=float, default=0.001,       help="Adam learning rate  (default 0.001)")
    p.add_argument("--val-frac",        type=float, default=0.20,        help="Val fraction from end of train  (default 0.20)")
    p.add_argument("--min-train",       type=int,   default=10,          help="Min train samples per split  (default 10)")
    p.add_argument("--min-test",        type=int,   default=5,           help="Min test samples per split  (default 5)")
    p.add_argument("--max-na-fraction", type=float, default=0.20,        help="Max NA fraction per probe  (default 0.20)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COMPARISON HELPERS — used only by __main__ in --model-type all mode
# ---------------------------------------------------------------------------

def _build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a per-model summary row from {model_type: summary_df}.

    Columns: model_type, top_probe, top_probe_auc,
             top10_median_auc, top10_median_accuracy
    """
    rows = []
    for mt, sdf in results.items():
        valid = sdf[sdf["N_runs"] > 0].copy()
        if valid.empty:
            rows.append({
                "model_type": mt, "top_probe": None, "top_probe_auc": None,
                "top10_median_auc": None, "top10_median_accuracy": None,
            })
            continue
        top10 = valid.head(10)
        rows.append({
            "model_type": mt,
            "top_probe":  valid.iloc[0]["Gene"],
            "top_probe_auc": (
                round(float(valid.iloc[0]["Median_TestAUC"]), 4)
                if pd.notna(valid.iloc[0]["Median_TestAUC"]) else None
            ),
            "top10_median_auc": (
                round(float(top10["Median_TestAUC"].median()), 4)
                if top10["Median_TestAUC"].notna().any() else None
            ),
            "top10_median_accuracy": (
                round(float(top10["Median_TestAcc"].median()), 4)
                if "Median_TestAcc" in top10.columns and top10["Median_TestAcc"].notna().any()
                else None
            ),
        })
    return pd.DataFrame(rows)


def _print_comparison_table(comp_df: pd.DataFrame) -> None:
    """Print comparison table to console."""
    print("\n" + "=" * 70)
    print("UNIVARIATE WRAPPER — MODEL COMPARISON")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("=" * 70)


def _plot_probe_auc_distribution(
    summary_df:     pd.DataFrame,
    gene_map:       "pd.Series",
    output_dir:     str,
    model_type:     str,
    disease_label:  str = "SONFH",
    control_label:  str = "control",
    annotate_top_n: int = 10,
    top_n_selected: int = None,
) -> str:
    """
    2-panel figure showing the distribution of per-probe Median_TestAUC across
    all screened probes.

    Panel A — Ranked AUC curve: all valid probes sorted descending by
    Median_TestAUC. Top N probes are annotated with 'probe_id — gene_symbol'.
    Shaded ribbon shows ±1 SD. A horizontal dashed line at AUC=0.9 marks the
    high-performance threshold.

    Panel B — Histogram: distribution of Median_TestAUC values across all
    probes. Vertical dashed line at 0.9 separates the high-AUC tail from the
    bulk.

    Saved as univariate_probe_auc_distribution.png in output_dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = (
        summary_df[
            (summary_df["N_runs"] > 0) & (summary_df["Median_TestAUC"].notna())
        ]
        .sort_values("Median_TestAUC", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    if valid.empty:
        print("_plot_probe_auc_distribution: no valid probe results — skipping")
        return ""

    gene_map_safe = gene_map if (gene_map is not None and len(gene_map)) else None
    n_mccv        = int(valid["N_runs"].iloc[0])
    aucs          = valid["Median_TestAUC"].values.astype(float)
    sds           = valid["SD_TestAUC"].fillna(0.0).values.astype(float)
    ranks         = np.arange(1, len(aucs) + 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: ranked curve ────────────────────────────────────────────────
    ax_a.plot(ranks, aucs, color="#1f77b4", linewidth=1.2, zorder=3)
    ax_a.fill_between(
        ranks, np.maximum(0, aucs - sds), np.minimum(1, aucs + sds),
        alpha=0.15, color="#1f77b4",
    )
    ax_a.axhline(0.9, color="#d62728", linewidth=0.8, linestyle="--", label="AUC = 0.90")
    ax_a.set_xlabel(f"Probe rank (n = {len(valid):,})", fontsize=10)
    ax_a.set_ylabel("Median Test AUC", fontsize=10)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlim(1, len(aucs))
    ax_a.set_title(
        f"Ranked Probe AUC — {model_type}\n"
        f"{disease_label} vs {control_label} | {n_mccv}× MCCV",
        fontsize=10,
    )
    ax_a.grid(True, alpha=0.25, linestyle="--")
    ax_a.legend(fontsize=9)

    # Annotate top N probes
    for i in range(min(annotate_top_n, len(valid))):
        probe = valid.loc[i, "Gene"]
        sym   = (gene_map_safe.get(probe, "") if gene_map_safe is not None else "")
        sym   = sym if sym and sym != "---" else ""
        label = f"{probe} — {sym}" if sym else probe
        ax_a.annotate(
            label,
            xy=(ranks[i], aucs[i]),
            xytext=(ranks[i] + max(len(aucs) * 0.015, 5), aucs[i]),
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "color": "#888", "lw": 0.5},
            va="center",
        )

    # ── Panel B: histogram ───────────────────────────────────────────────────
    n_high = int((aucs >= 0.9).sum())
    ax_b.hist(aucs, bins=40, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax_b.axvline(0.9, color="#d62728", linewidth=1.0, linestyle="--",
                 label=f"AUC ≥ 0.90 ({n_high} probes)")
    ax_b.set_xlabel("Median Test AUC", fontsize=10)
    ax_b.set_ylabel("Number of probes", fontsize=10)
    ax_b.set_title(
        f"AUC Distribution — {len(valid):,} probes\n"
        f"{disease_label} vs {control_label}",
        fontsize=10,
    )
    ax_b.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_b.legend(fontsize=9)

    top_n_label = f"  |  top_{top_n_selected} selected" if top_n_selected else ""
    plt.suptitle(
        f"Univariate ANN probe screening — {disease_label} vs {control_label}{top_n_label}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    fname    = f"univariate_probe_auc_distribution_top{top_n_selected}.png" if top_n_selected else "univariate_probe_auc_distribution.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Probe AUC distribution figure saved: {out_path}")
    return out_path


def _plot_comparison_composite(
    results: dict,
    output_dir: str,
) -> str:
    """
    4-panel composite PNG comparing model performance.

    Panel A : Top-20 Median_TestAUC curves per model
    Panel B : Top-20 Median_TestAcc curves per model
    Panel C : Top-20 probe overlap counts between model pairs
    Panel D : Median_TestAUC box distribution per model (full probe set)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_comparison_composite.png")

    model_names = list(results.keys())
    colors = ["#2196F3", "#4CAF50", "#FF5722"][:len(model_names)]
    top_n = 20

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- Panel A: Top-20 Median_TestAUC ----
    for i, (mt, sdf) in enumerate(results.items()):
        valid = sdf[sdf["N_runs"] > 0]
        vals = valid["Median_TestAUC"].dropna().head(top_n).reset_index(drop=True)
        ax_a.plot(range(1, len(vals) + 1), vals, marker="o", ms=4,
                  label=mt, color=colors[i], linewidth=1.5)
    ax_a.set_title(f"Top {top_n} Median Test AUC", fontsize=10)
    ax_a.set_xlabel("Probe rank", fontsize=8)
    ax_a.set_ylabel("Median TestAUC", fontsize=8)
    ax_a.legend(fontsize=7)
    ax_a.grid(alpha=0.3, linestyle="--")

    # ---- Panel B: Top-20 Median_TestAcc ----
    for i, (mt, sdf) in enumerate(results.items()):
        valid = sdf[sdf["N_runs"] > 0]
        if "Median_TestAcc" not in valid.columns:
            continue
        vals = valid["Median_TestAcc"].dropna().head(top_n).reset_index(drop=True)
        ax_b.plot(range(1, len(vals) + 1), vals, marker="o", ms=4,
                  label=mt, color=colors[i], linewidth=1.5)
    ax_b.set_title(f"Top {top_n} Median Test Accuracy", fontsize=10)
    ax_b.set_xlabel("Probe rank", fontsize=8)
    ax_b.set_ylabel("Median TestAcc", fontsize=8)
    ax_b.legend(fontsize=7)
    ax_b.grid(alpha=0.3, linestyle="--")

    # ---- Panel C: Top-20 probe overlap counts ----
    top20_sets = {
        mt: set(sdf[sdf["N_runs"] > 0]["Gene"].head(top_n).tolist())
        for mt, sdf in results.items()
    }
    if len(model_names) >= 2:
        pair_labels, pair_counts = [], []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                pair_labels.append(f"{a[:3]}∩{b[:3]}")
                pair_counts.append(len(top20_sets[a] & top20_sets[b]))
        if len(model_names) == 3:
            all3 = (top20_sets[model_names[0]]
                    & top20_sets[model_names[1]]
                    & top20_sets[model_names[2]])
            pair_labels.append("all 3")
            pair_counts.append(len(all3))
        bar_colors = ["#9C27B0", "#009688", "#FF9800", "#607D8B"][:len(pair_labels)]
        bars = ax_c.bar(pair_labels, pair_counts, color=bar_colors)
        for bar, cnt in zip(bars, pair_counts):
            ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                      str(cnt), ha="center", va="bottom", fontsize=9)
        ax_c.set_ylim(0, top_n + 2)
    ax_c.set_title(f"Top-{top_n} Probe Overlap", fontsize=10)
    ax_c.set_ylabel("Shared probes", fontsize=8)
    ax_c.tick_params(axis="x", labelsize=8)
    ax_c.grid(axis="y", alpha=0.3, linestyle="--")

    # ---- Panel D: Median_TestAUC box distribution (full probe set per model) ----
    data, labels = [], []
    for mt, sdf in results.items():
        valid = sdf[sdf["N_runs"] > 0]
        aucs = valid["Median_TestAUC"].dropna().values
        if len(aucs) > 0:
            data.append(aucs)
            labels.append(mt)
    if data:
        bp = ax_d.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
    ax_d.set_title("Median TestAUC Distribution (all probes)", fontsize=10)
    ax_d.set_ylabel("Median TestAUC", fontsize=8)
    ax_d.tick_params(axis="x", labelsize=8)
    ax_d.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Univariate Wrapper — Model Comparison", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    args = parse_args()

    models = (
        ["professor_ann", "logistic", "svm_rbf"]
        if args.model_type == "all"
        else [args.model_type]
    )

    # Kwargs shared by every model run
    _pipeline_kwargs = dict(
        preprocessed_csv=args.input,
        soft_gz_path=args.soft,
        disease_label=args.disease_label,
        control_label=args.control_label,
        use_filter=not args.no_filter,
        filter_top_n=args.filter_top_n,
        n_mccv=args.n_mccv,
        train_frac=args.train_frac,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        val_frac=args.val_frac,
        min_train=args.min_train,
        min_test=args.min_test,
        max_na_fraction=args.max_na_fraction,
        # top100 = PRIMARY Weka output; top500 = optional exploratory only
        weka_top_ns=[100, 500],
        relation_name="univariate_ann",
    )

    if len(models) == 1:
        # Single-model path — existing behavior, output goes to args.outdir directly
        run_univariate_pipeline(
            output_dir=args.outdir,
            model_type=models[0],
            **_pipeline_kwargs,
        )
    else:
        # Comparison mode — each model gets a subdir, comparison files at base
        results = {}
        for model_type in models:
            out_dir_m = os.path.join(args.outdir, model_type)
            print(f"\n{'='*60}\n  Running: {model_type}\n{'='*60}")
            run_univariate_pipeline(
                output_dir=out_dir_m,
                model_type=model_type,
                **_pipeline_kwargs,
            )
            summary_path = os.path.join(out_dir_m, "wrapper_summary_per_gene.csv")
            results[model_type] = pd.read_csv(summary_path)

        # Comparison table
        comp_df = _build_comparison_table(results)
        _print_comparison_table(comp_df)
        comp_path = os.path.join(args.outdir, "model_comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparison table saved: {comp_path}")

        # Composite PNG
        png_path = _plot_comparison_composite(results, output_dir=args.outdir)
        print(f"Composite PNG saved:    {png_path}")
