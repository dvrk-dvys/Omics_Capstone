"""
preprocess.py — Filter and normalize the pseudobulk matrix for Weka

WHY THIS SCRIPT EXISTS:
  pseudobulk.py produced a 5 × 33,538 gene matrix — but most of those genes
  are useless noise (zero or near-zero counts in nearly every sample).
  This script does two things:

  1. FILTER — remove genes that aren't meaningfully expressed.
     Keeps only genes with a count > 0 in at least 2 of the 5 samples.
     Rationale: if a gene is silent in 4/5 patients it can't help a classifier
     distinguish ONFH from OA.

  2. NORMALIZE — correct for the fact that different samples have different
     total counts (onfh_1 has 62M counts, onfh_2 has only 23M).
     Without normalization, a gene could appear "highly expressed" in onfh_1
     just because that sample has more total data — not because of biology.
     We use log-CPM: convert to Counts Per Million, then apply log(x+1).

OUTPUT:
  A filtered, normalized CSV still in Weka format:
  5 rows × (filtered genes + class column)
  This file feeds directly into file_splitter.py and Weka.

NOTE ON TRANSPOSE:
  Unlike bulk RNA-seq, our pseudobulk.py already output samples as rows and
  genes as columns — the correct Weka orientation. transpose.py is NOT needed
  for this dataset. Going straight from this script to file_splitter.py is correct.

Usage:
  python3 preprocess.py [<pseudobulk_csv> <output_csv>]
  python3 preprocess.py  (uses defaults: data/pseudobulk/pseudobulk_matrix.csv → data/pseudobulk/preprocessed_matrix.csv)
  python3 preprocess.py data/pseudobulk/pseudobulk_matrix.csv data/pseudobulk/preprocessed_matrix.csv
"""

import sys
import os
import pandas as pd
import numpy as np


def load_pseudobulk(path: str) -> pd.DataFrame:
    """
    Load the pseudobulk CSV produced by pseudobulk.py.

    Reads the CSV back in with the sample names as the row index.
    Separates the gene expression columns from the class label column
    so we can work on them independently.

    Parameters:
      path : str — path to pseudobulk_matrix.csv

    Returns:
      pd.DataFrame — shape (5, 33539): gene columns + class column
    """
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded: {df.shape[0]} samples x {df.shape[1]} columns "
          f"({df.shape[1] - 1} genes + class)")
    return df


def filter_genes(df: pd.DataFrame, min_samples: int = 2) -> pd.DataFrame:
    """
    Remove genes that are not expressed in enough samples.

    What "expressed" means here: count > 0 in a given sample.
    A gene that is zero in 4 out of 5 samples carries no useful signal
    for distinguishing ONFH from OA — it just adds noise and slows Weka down.

    Strategy:
      For each gene column, count how many of the 5 samples have count > 0.
      Keep the gene only if that count >= min_samples (default: 2).

    Parameters:
      df          : pd.DataFrame — pseudobulk matrix with class column last
      min_samples : int — minimum number of samples that must express a gene to keep it

    Returns:
      pd.DataFrame — same shape minus the dropped gene columns, class column preserved
    """
    # Split gene columns from class column (always keep class)
    gene_cols = df.columns[:-1]
    class_col = df["class"]

    gene_df = df[gene_cols]

    # For each gene: count how many samples have count > 0
    expressed_in_n_samples = (gene_df > 0).sum(axis=0)

    # Keep genes expressed in at least min_samples samples
    keep_mask = expressed_in_n_samples >= min_samples
    filtered = gene_df.loc[:, keep_mask]

    n_removed = (~keep_mask).sum()
    n_kept    = keep_mask.sum()
    print(f"\nFiltering (expressed in >= {min_samples} samples):")
    print(f"  Genes before : {len(gene_cols)}")
    print(f"  Genes removed: {n_removed}  (zero or near-zero in most samples)")
    print(f"  Genes kept   : {n_kept}")

    # Re-attach class column at the end
    filtered["class"] = class_col
    return filtered


def normalize_log_cpm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize gene counts using log-CPM (log Counts Per Million).

    WHY NORMALIZE:
      Different samples have different total counts:
        onfh_1:  62,450,342  total counts
        onfh_2:  23,286,947  total counts  ← less than 3x onfh_1
        oa_2:    70,496,767  total counts
      Without correction, a gene with 1000 counts in onfh_2 looks less active
      than the same gene with 1000 counts in onfh_1, even though relative to
      each sample's total it's the same. That's a technical artifact, not biology.

    HOW LOG-CPM WORKS:
      Step 1 — CPM: divide each gene's count by the sample's total counts, multiply by 1,000,000
               This puts all samples on the same scale (per-million reads).
      Step 2 — log1p: apply log(count + 1) to compress the range.
               Raw counts span 0 to millions; log brings that to 0–15 range.
               The +1 avoids log(0) which is undefined.

    Parameters:
      df : pd.DataFrame — filtered matrix with class column last

    Returns:
      pd.DataFrame — same shape, gene values replaced with log-CPM values
    """
    gene_cols = df.columns[:-1]
    class_col = df["class"]
    gene_df   = df[gene_cols].copy().astype(float)

    # Step 1: divide each row by its total, multiply by 1,000,000
    row_totals    = gene_df.sum(axis=1)           # total counts per sample
    cpm           = gene_df.div(row_totals, axis=0) * 1_000_000

    # Step 2: log(x + 1) to compress the scale
    log_cpm       = np.log1p(cpm)

    print(f"\nNormalization (log-CPM):")
    print(f"  Raw count range  : {gene_df.values.min():.0f} – {gene_df.values.max():,.0f}")
    print(f"  log-CPM range    : {log_cpm.values.min():.3f} – {log_cpm.values.max():.3f}")
    print(f"  Sample totals (pre-norm):")
    for sample, total in row_totals.items():
        print(f"    {sample}: {total:,.0f}")

    # Re-attach class column at the end
    log_cpm["class"] = class_col
    return log_cpm


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a final summary of the preprocessed matrix before saving.
    Confirms shape, gene count, and Weka readiness.

    Parameters:
      df : pd.DataFrame — the fully preprocessed matrix
    """
    print("\n" + "=" * 60)
    print("PREPROCESSED MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Gene columns   : {df.shape[1] - 1}")
    print(f"Class column   : '{df.columns[-1]}'  (last — correct for Weka)")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())
    print(f"\nPreview (first 4 genes, all samples):")
    print(df.iloc[:, :4].round(3))
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run file_splitter.py on this output (for Weka single-gene wrapper)")
    print("  2. Load full matrix directly into Weka Explorer")
    print("  NOTE: transpose.py is NOT needed — samples are already rows (Weka format)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
DEFAULT_INPUT_CSV  = "/Users/jordanharris/Code/Omics_Capstone/data/pseudobulk/pseudobulk_matrix.csv"
DEFAULT_OUTPUT_CSV = "/Users/jordanharris/Code/Omics_Capstone/data/pseudobulk/preprocessed_matrix.csv"

if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_path  = DEFAULT_INPUT_CSV
        output_path = DEFAULT_OUTPUT_CSV
    elif len(sys.argv) == 3:
        input_path  = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print("Usage: python3 preprocess.py [<pseudobulk_csv> <output_csv>]")
        print(f"  Default input  : {DEFAULT_INPUT_CSV}")
        print(f"  Default output : {DEFAULT_OUTPUT_CSV}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load pseudobulk matrix
    df = load_pseudobulk(input_path)

    # Step 2: Filter lowly expressed genes
    df = filter_genes(df, min_samples=2)

    # Step 3: Normalize with log-CPM
    df = normalize_log_cpm(df)

    # Step 4: Summary check
    print_summary(df)

    # Step 5: Save
    df.to_csv(output_path)
    print(f"\nSaved to: {output_path}")
