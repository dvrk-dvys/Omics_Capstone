"""
preprocess.py — Filter the parsed microarray matrix for Weka

WHY THIS SCRIPT EXISTS:
  parse_series_matrix.py produced a 40 × ~49,000 probe matrix — but the vast
  majority of those probes carry no useful discriminative signal (flat expression
  across all samples, or near-identical values regardless of condition).
  This script filters them out so feature_select.py and Weka work on a cleaner
  and smaller feature set.

  NOTE: Unlike the old scRNA-seq pipeline, NO normalization is applied here.
  Affymetrix microarray data in the GEO series matrix has already been processed
  (RMA normalization produces log2 intensity values that are directly comparable
  across all 40 samples). Applying log-CPM on top of log2 values would be wrong.

  WHAT THIS SCRIPT DOES:
  1. FILTER — remove low-variance probes.
     Keeps only probes whose interquartile range (IQR) across all 40 samples
     exceeds a threshold (default: IQR > 0.2 in log2 space).
     Rationale: a probe with IQR < 0.2 shows less than 0.2 log2-units of
     spread — it is essentially flat and cannot help distinguish SONFH from
     control regardless of any classifier.

OUTPUT:
  A filtered CSV in Weka format:
  40 rows × (filtered probes + class column)
  This file feeds directly into feature_select.py and Weka.

Usage:
  python3 preprocess.py [<parsed_csv> <output_csv>]
  python3 preprocess.py  (uses defaults shown below)
  python3 preprocess.py data/femoral_head_necrosis/parsed/parsed_matrix.csv data/.../preprocessed_matrix.csv
"""

import sys
import os
import pathlib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# PATHS  (relative to app/ — works from any working directory)
# ---------------------------------------------------------------------------
_APP_DIR   = pathlib.Path(__file__).resolve().parent.parent
INPUT_CSV  = str(_APP_DIR / "data" / "output" / "parsed" / "parsed_matrix.csv")
OUTPUT_CSV = str(_APP_DIR / "data" / "output" / "parsed" / "preprocessed_matrix.csv")

# ---------------------------------------------------------------------------
# WEKA STANDALONE PATHS  (used only when running this file directly)
# ---------------------------------------------------------------------------
_PROJECT_ROOT    = _APP_DIR.parent.parent   # /Omics_Capstone/
_WEKA_INPUT_CSV  = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "parsed" / "parsed_matrix.csv")
_WEKA_OUTPUT_CSV = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "parsed" / "preprocessed_matrix.csv")


def load_parsed(path: str) -> pd.DataFrame:
    """
    Load the parsed CSV produced by parse_series_matrix.py.

    Parameters:
      path : str — path to parsed_matrix.csv

    Returns:
      pd.DataFrame — shape (40, N_probes + 1): probe columns + class column
    """
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded: {df.shape[0]} samples × {df.shape[1]} columns "
          f"({df.shape[1] - 1} probes + class)")
    return df


def filter_probes(df: pd.DataFrame, iqr_threshold: float = 0.2) -> pd.DataFrame:
    """
    Remove low-variance probes using Interquartile Range (IQR).

    WHY IQR INSTEAD OF ZERO-COUNT FILTER:
      Microarray data (log2 RMA values) almost never has true zeros —
      every probe gets a background-corrected signal even if the gene isn't
      expressed. So the old "count > 0 in N samples" filter doesn't apply.
      Instead we use IQR: a probe that barely moves across all 40 samples
      (IQR < 0.2 log2 units) contributes no information and is removed.

    Parameters:
      df            : pd.DataFrame — parsed matrix with class column last
      iqr_threshold : float — minimum IQR to keep a probe (default 0.2 log2 units)

    Returns:
      pd.DataFrame — same shape minus dropped probe columns, class column preserved
    """
    probe_cols = df.columns[:-1]
    class_col  = df["class"]
    probe_df   = df[probe_cols]

    q75 = probe_df.quantile(0.75, axis=0)
    q25 = probe_df.quantile(0.25, axis=0)
    iqr = q75 - q25

    keep_mask = iqr >= iqr_threshold
    filtered  = probe_df.loc[:, keep_mask]

    n_removed = (~keep_mask).sum()
    n_kept    = keep_mask.sum()
    print(f"\nFiltering (IQR >= {iqr_threshold} log2 units across all samples):")
    print(f"  Probes before  : {len(probe_cols)}")
    print(f"  Probes removed : {n_removed}  (flat expression — no discriminative signal)")
    print(f"  Probes kept    : {n_kept}")

    filtered["class"] = class_col
    return filtered


def check_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify the data looks like log2 microarray values and report the range.

    Microarray data from GEO series matrix (RMA processed) is already in log2
    space. Values should roughly be in the range 2–16. No transformation applied.

    Parameters:
      df : pd.DataFrame — filtered matrix with class column last

    Returns:
      pd.DataFrame — unchanged (this is a check, not a transform)
    """
    probe_cols = df.columns[:-1]
    vals = df[probe_cols].values.flatten()

    print(f"\nNormalization check (data is already log2 from RMA — no transform applied):")
    print(f"  Value range : {vals.min():.3f} – {vals.max():.3f}")
    print(f"  Mean        : {vals.mean():.3f}")
    print(f"  Std dev     : {vals.std():.3f}")

    if vals.max() > 25:
        print("  WARNING: values exceed 25 — data may NOT be log2 transformed. "
              "Check the series matrix source.")
    else:
        print("  OK: range consistent with log2 RMA microarray values.")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a final summary of the preprocessed matrix before saving.

    Parameters:
      df : pd.DataFrame — the fully preprocessed matrix
    """
    print("\n" + "=" * 60)
    print("PREPROCESSED MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Probe columns  : {df.shape[1] - 1}")
    print(f"Class column   : '{df.columns[-1]}'  (last — correct for Weka)")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())
    print(f"\nPreview (first 4 probes, all samples):")
    print(df.iloc[:, :4].round(3))
    print("=" * 60)
    print("\nNext step:")
    print("  python3 feature_select.py")




# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_path  = _WEKA_INPUT_CSV
        output_path = _WEKA_OUTPUT_CSV
    elif len(sys.argv) == 3:
        input_path  = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print("Usage: python3 preprocess.py [<parsed_csv> <output_csv>]")
        print(f"  Default input  : {_WEKA_INPUT_CSV}")
        print(f"  Default output : {_WEKA_OUTPUT_CSV}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load parsed matrix from parse_series_matrix.py
    df = load_parsed(input_path)

    # Step 2: Filter low-variance probes (IQR-based)
    df = filter_probes(df, iqr_threshold=0.2)

    # Step 3: Verify normalization (data is already log2 — no transform applied)
    df = check_normalization(df)

    # Step 4: Summary check
    print_summary(df)

    # Step 5: Save
    df.to_csv(output_path)
    print(f"\nSaved to: {output_path}")
