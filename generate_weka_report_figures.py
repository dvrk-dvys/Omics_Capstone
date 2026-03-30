"""
generate_report_figures.py — Report figure generator for the SONFH capstone.

Run this script AFTER both pipelines have been executed:
  1. Weka pipeline  → python omics_ml_pipeline/app/utils/feature_select.py
  2. Python pipeline → python omics_ml_pipeline/app/main.py --config config/pipeline.yaml

What this script produces
─────────────────────────
  report/figures/
  ├── weka_model_comparison.csv       ← Weka results in the same schema as Python's CSV
  ├── fig_weka_model_comparison.png   ← Weka classifier bar chart
  └── fig_python_model_comparison.png ← Python classifier bar chart

The EDA composite figures (fig_eda_composite) are generated automatically by
each pipeline run and saved alongside the other EDA plots:
  data/femoral_head_necrosis/EDA/eda_composite.png          ← Weka EDA composite
  omics_ml_pipeline/app/data/output/plots/eda_composite.png ← Python EDA composite

Usage
─────
  python generate_report_figures.py

  Run from the project root:  /Users/jordanharris/Code/Omics_Capstone/
"""

import os
import re
import glob
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_ROOT         = pathlib.Path(__file__).resolve().parent

WEKA_MODELS   = _ROOT / "data" / "femoral_head_necrosis" / "weka_models"
PYTHON_CSV    = _ROOT / "omics_ml_pipeline" / "app" / "data" / "output" / "models" / "model_comparison.csv"
REPORT_DIR    = _ROOT / "report" / "figures"

WEKA_OUT_CSV  = REPORT_DIR / "weka_model_comparison.csv"
WEKA_CHART    = REPORT_DIR / "fig_weka_model_comparison.png"
PYTHON_CHART  = REPORT_DIR / "fig_python_model_comparison.png"

# ---------------------------------------------------------------------------
# STEP 1 — Parse Weka .txt files → weka_model_comparison.csv
# ---------------------------------------------------------------------------

# Human-readable labels keyed by filename stem
_WEKA_LABELS = {
    "auto_weka":                      "Auto-Weka (PART)",
    "j48_tree":                       "J48",
    "randomforest":                   "Random Forest",
    "naive_bayes":                    "Naive Bayes",
    "multilayerpreceptron":           "MLP",
    "functions_smo":                  "SMO (SVM)",
    "lazy_ibk":                       "IBk (k=1)",
    "lazy_ibk_knn_3":                 "IBk (k=3)",
    "lazy_ibk_knn_5":                 "IBk (k=5)",
    "attribute_selection_randomforest": None,   # attribute selection only — no classifier metrics
}


def parse_weka_results(weka_models_dir: pathlib.Path) -> pd.DataFrame:
    """
    Read every Weka classifier .txt result file and extract:
      - accuracy_mean   (standard accuracy, 0–1)
      - f1_weighted_mean
      - roc_auc_mean

    Returns a DataFrame in the same column schema as Python's model_comparison.csv
    so that the same plotting function can handle both.

    Weka 3.8 output format (numbers may use ',' or '.' as decimal separator):
      Correctly Classified Instances  N  X.X %
      Weighted Avg.  TP  FP  Precision  Recall  F-Measure  MCC  ROC  PRC
    """
    rows = []
    for fpath in sorted(glob.glob(str(weka_models_dir / "*.txt"))):
        stem  = pathlib.Path(fpath).stem
        label = _WEKA_LABELS.get(stem)
        if label is None:
            continue                        # skip attribute-selection-only files

        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        acc_m = re.search(
            r"Correctly Classified Instances\s+\d+\s+([\d.]+)\s+%", text
        )
        wt_m  = re.search(r"Weighted Avg\.\s+(.*)", text)

        if not (acc_m and wt_m):
            print(f"  [skip] {stem} — could not locate metrics")
            continue

        acc   = float(acc_m.group(1)) / 100.0
        parts = wt_m.group(1).split()
        nums  = []
        for p in parts:
            try:
                nums.append(float(p.replace(",", ".")))
            except ValueError:
                pass

        # Weighted Avg columns: TP, FP, Precision, Recall, F-Measure, MCC, ROC, PRC
        if len(nums) < 7:
            print(f"  [skip] {stem} — weighted-avg line has fewer columns than expected")
            continue

        rows.append({
            "model":                 label,
            "source":                "weka_10fold_cv",
            "accuracy_mean":         round(acc,      4),
            "roc_auc_mean":          round(nums[6],  4),   # ROC Area
            "f1_weighted_mean":      round(nums[4],  4),   # F-Measure
            "balanced_accuracy_mean": None,                 # not reported by Weka
        })

    if not rows:
        raise RuntimeError(f"No parseable Weka result files found in {weka_models_dir}")

    df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    print(f"  Parsed {len(df)} Weka models")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Shared model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    df:          pd.DataFrame,
    output_path: pathlib.Path,
    title:       str,
    primary_metric: str = "roc_auc_mean",
) -> None:
    """
    Grouped horizontal bar chart comparing classifier performance.

    Works with both the Weka CSV (columns: accuracy_mean, roc_auc_mean, f1_weighted_mean)
    and the Python CSV (same columns plus balanced_accuracy_mean).

    For the Python pipeline, balanced_accuracy_mean is plotted instead of accuracy_mean
    because it is the more meaningful metric on an imbalanced dataset (30:10).
    For the Weka pipeline, accuracy_mean is plotted because Weka only reports
    standard accuracy.
    """
    has_balanced = (
        "balanced_accuracy_mean" in df.columns
        and df["balanced_accuracy_mean"].notna().any()
    )

    if has_balanced:
        metrics  = ["roc_auc_mean", "balanced_accuracy_mean", "f1_weighted_mean"]
        labels   = ["ROC AUC", "Balanced Accuracy", "F1 (weighted)"]
        note     = "Balanced accuracy used (accounts for 30:10 class imbalance)"
    else:
        metrics  = ["roc_auc_mean", "accuracy_mean", "f1_weighted_mean"]
        labels   = ["ROC AUC", "Accuracy", "F1 (weighted)"]
        note     = "Standard accuracy (10-fold CV)"

    colours = ["#c0392b", "#2c5f8a", "#e8a838"]

    # Sort ascending so best model is at top when plotted as barh
    df = df.sort_values(primary_metric, ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))
    bar_h = 0.24

    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.82)))

    for i, (col, lab, col_colour) in enumerate(zip(metrics, labels, colours)):
        if col not in df.columns or df[col].isna().all():
            continue
        offset = (i - 1) * bar_h
        bars = ax.barh(
            y + offset, df[col], bar_h,
            label=lab, color=col_colour, alpha=0.85, edgecolor="white",
        )
        for bar, val in zip(bars, df[col]):
            if pd.notna(val):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7.5, color="#333")

    # Highlight the best model row
    best_idx = df[primary_metric].idxmax()
    ax.axhspan(best_idx - bar_h * 2.2, best_idx + bar_h * 2.2,
               color="#d4edda", alpha=0.45, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1.14)
    ax.axvline(0.9, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    ax.text(0.01, -0.06, note, fontsize=8, color="#666", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {REPORT_DIR}\n")

    # ── Step 1: Parse Weka .txt files → weka_model_comparison.csv ──────────
    print("[ 1/3 ] Parsing Weka result files...")
    if not WEKA_MODELS.exists():
        print(f"  ERROR: Weka models directory not found: {WEKA_MODELS}")
        print("  Run feature_select.py first to generate Weka results.")
    else:
        weka_df = parse_weka_results(WEKA_MODELS)
        weka_df.to_csv(WEKA_OUT_CSV, index=False)
        print(f"  Saved: {WEKA_OUT_CSV}\n")

        # ── Step 2: Weka model comparison bar chart ─────────────────────────
        print("[ 2/3 ] Generating Weka model comparison chart...")
        plot_model_comparison(
            df          = weka_df,
            output_path = WEKA_CHART,
            title       = (
                "Weka Classifier Performance — 10-fold Cross-Validation\n"
                "GSE123568 | Top 100 Features | 40 Samples (30 SONFH / 10 Control)"
            ),
        )
        print()

    # ── Step 3: Python model comparison bar chart ───────────────────────────
    print("[ 3/3 ] Generating Python pipeline model comparison chart...")
    if not PYTHON_CSV.exists():
        print(f"  ERROR: Python model comparison CSV not found: {PYTHON_CSV}")
        print("  Run main.py first to generate Python pipeline results.")
    else:
        python_df = pd.read_csv(PYTHON_CSV)

        # Clean up model names for readability
        python_df["model"] = (
            python_df["model"]
            .str.replace("baseline_", "", regex=False)
            .str.replace("tuned_", "Tuned — ", regex=False)
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        plot_model_comparison(
            df          = python_df,
            output_path = PYTHON_CHART,
            title       = (
                "Python Pipeline Classifier Performance — Stratified 5-fold CV\n"
                "GSE123568 | Top 50 Features | 40 Samples (30 SONFH / 10 Control) | MLflow run r024"
            ),
        )
        print()

    print("Done.")
    print(f"\nFiles written to {REPORT_DIR}:")
    for f in sorted(REPORT_DIR.glob("*")):
        print(f"  {f.name}")
