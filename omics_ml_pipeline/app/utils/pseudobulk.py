"""
pseudobulk.py — Collapse scRNA-seq cell-level data into one row per patient sample

WHY THIS SCRIPT EXISTS:
  The raw scRNA-seq data has one row per CELL (~46,000 cells across 5 patients).
  Weka needs one row per PATIENT SAMPLE (5 rows total).
  This script bridges that gap by summing all cell counts within each sample
  into a single representative row — a technique called "pseudobulk aggregation."

WHAT IT READS (per sample folder):
  barcodes.tsv.gz  — list of cell IDs (one per row, tells us how many cells there are)
  features.tsv.gz  — list of genes (one per row: Ensembl ID, gene name, type)
  matrix.mtx.gz    — the actual counts in sparse format (gene#, cell#, count triplets)

WHAT IT PRODUCES:
  data/pseudobulk/pseudobulk_matrix.csv — 5 rows × 33,539 columns (33,538 genes + 1 class label column)
  Each row = one patient sample, summed across all its cells.
  This CSV is the input to normalize → transpose → file_splitter → Weka.

SAMPLE LAYOUT:
  GSM9463148_onfh_1  →  Steroid-induced ONFH, patient 1  (8,160 cells)
  GSM9463149_onfh_2  →  Steroid-induced ONFH, patient 2  (2,128 cells)
  GSM9463150_oa_1    →  Hip Osteoarthritis, patient 1    (8,164 cells)
  GSM9463151_oa_2    →  Hip Osteoarthritis, patient 2    (16,489 cells)
  GSM9463152_oa_3    →  Hip Osteoarthritis, patient 3    (11,950 cells)

Usage:
  python3 pseudobulk.py [<samples_dir> <output_csv>]
  python3 pseudobulk.py  (uses defaults: data/femoral_head_necrosis → data/pseudobulk/pseudobulk_matrix.csv)
  python3 pseudobulk.py data/femoral_head_necrosis data/pseudobulk/pseudobulk_matrix.csv
"""

import sys
import os
import gzip
import scipy.io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Sample metadata
# Maps each folder name to its class label (ONFH or OA).
# This is the ground truth for what condition each patient has.
# ---------------------------------------------------------------------------
SAMPLE_LABELS = {
    "GSM9463148_onfh_1": "ONFH",
    "GSM9463149_onfh_2": "ONFH",
    "GSM9463150_oa_1":   "OA",
    "GSM9463151_oa_2":   "OA",
    "GSM9463152_oa_3":   "OA",
}


def load_sample(sample_dir: str) -> pd.Series:
    """
    Load one patient sample's 3 files and return a single row of gene counts.

    What this does step by step:
      1. Finds the barcodes, features, and matrix files inside the sample folder.
      2. Reads the matrix.mtx.gz (sparse format) into a dense genes × cells table.
      3. Reads the features.tsv.gz to get human-readable gene names.
      4. Sums across all cells (axis=1) → one total count per gene.
      5. Returns a pandas Series indexed by gene name: one number per gene.

    Why we sum (not average):
      Summing preserves the total signal from all cells. For the pseudobulk
      approach, summed counts per gene behave like bulk RNA-seq counts and can
      be fed into standard normalization methods downstream.

    Parameters:
      sample_dir : str — path to a single sample folder containing the 3 .gz files

    Returns:
      pd.Series — index = gene names (33,538 entries), values = summed raw counts
    """

    # --- Locate the 3 required files inside the sample folder ---
    # We search by suffix rather than hardcoding full names because the
    # filename prefix differs per sample (e.g. GSM9463148_onfh_1_matrix.mtx.gz)
    files = os.listdir(sample_dir)

    def find_file(suffix):
        matches = [f for f in files if f.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"No file ending in '{suffix}' found in {sample_dir}")
        return os.path.join(sample_dir, matches[0])

    barcodes_path = find_file("barcodes.tsv.gz")
    features_path = find_file("features.tsv.gz")
    matrix_path   = find_file("matrix.mtx.gz")

    # --- Read barcodes (cell IDs) ---
    # Each line is one cell barcode, e.g. "AAACCCAAGCGACATG-1"
    # We only use this to confirm cell count — not needed for pseudobulk itself
    with gzip.open(barcodes_path, "rt") as f:
        barcodes = [line.strip() for line in f]
    n_cells = len(barcodes)

    # --- Read features (gene list) ---
    # Tab-separated: Ensembl ID | gene name | feature type
    # We use column 1 (gene name, e.g. "TP53") as our row labels
    # If gene names have duplicates, we fall back to Ensembl IDs (column 0)
    with gzip.open(features_path, "rt") as f:
        feature_rows = [line.strip().split("\t") for line in f]
    gene_names = [row[1] for row in feature_rows]   # human-readable names

    # --- Read the sparse matrix ---
    # matrix.mtx.gz is in MatrixMarket format: each line = (gene_index, cell_index, count)
    # scipy.io.mmread handles this format natively
    # Result: a sparse matrix of shape (n_genes, n_cells)
    with gzip.open(matrix_path, "rb") as f:
        sparse_matrix = scipy.io.mmread(f)          # shape: genes × cells

    # --- Convert sparse → dense and sum across all cells ---
    # .toarray() turns the sparse representation into a full numpy matrix
    # .sum(axis=1) adds across columns (cells), leaving one number per gene row
    # .flatten() removes the extra dimension so we get a 1D array
    dense = sparse_matrix.toarray()                 # shape: (33538, n_cells)
    gene_totals = dense.sum(axis=1).flatten()       # shape: (33538,)

    # --- Package as a named Series ---
    # Index = gene names, so when we stack all samples the columns align correctly
    sample_series = pd.Series(gene_totals, index=gene_names, dtype=float)

    sample_name = os.path.basename(sample_dir)
    print(f"  Loaded: {sample_name}  |  {n_cells} cells  |  "
          f"{(gene_totals > 0).sum()} genes with any expression")

    return sample_series


def build_pseudobulk_matrix(samples_dir: str) -> pd.DataFrame:
    """
    Loop over all 5 sample folders and stack them into a single DataFrame.

    What this does:
      For each sample folder listed in SAMPLE_LABELS, calls load_sample() to
      get one row of gene counts, then stacks all 5 rows into a DataFrame.
      Finally, appends a 'class' column at the end with the condition label
      (ONFH or OA) for each sample row.

    Why class goes last:
      Weka requires the class/label column to be the final column in the file.
      Transpose.py and file_splitter.py also rely on this convention.

    Parameters:
      samples_dir : str — path to the folder containing all 5 sample subfolders

    Returns:
      pd.DataFrame — shape (5, 33539): 5 rows × 33538 gene columns + 1 class column
    """

    rows = {}   # dict of { sample_name: pd.Series of gene counts }

    print("\nLoading all 5 samples...\n")
    for sample_name, label in SAMPLE_LABELS.items():
        sample_path = os.path.join(samples_dir, sample_name)
        if not os.path.isdir(sample_path):
            raise FileNotFoundError(
                f"Expected sample folder not found: {sample_path}\n"
                f"Make sure you're pointing at the femoral_head_necrosis directory."
            )
        rows[sample_name] = load_sample(sample_path)

    # Stack all 5 Series into a DataFrame (each Series becomes one row)
    # Index = sample names, columns = gene names
    df = pd.DataFrame(rows).T    # .T because pd.DataFrame(dict) puts keys as columns

    # Append the class label as the final column (required by Weka)
    df["class"] = [SAMPLE_LABELS[name] for name in df.index]

    return df


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the pseudobulk matrix so you can sanity-check the output
    before saving. Shows shape, class distribution, and a preview of the data.

    Parameters:
      df : pd.DataFrame — the completed pseudobulk matrix (5 rows × 33539 cols)
    """

    print("\n" + "=" * 60)
    print("PSEUDOBULK MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape            : {df.shape[0]} rows (samples) x {df.shape[1]} columns")
    print(f"Gene columns     : {df.shape[1] - 1}")
    print(f"Class column     : '{df.columns[-1]}'  (last column — correct for Weka)")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())
    print(f"\nSample names (row index):")
    for name in df.index:
        print(f"  {name}  →  {df.loc[name, 'class']}")
    print(f"\nPreview (first 3 genes, all samples):")
    print(df.iloc[:, :3])
    print(f"\nTotal counts per sample (should all be > 0):")
    print(df.iloc[:, :-1].sum(axis=1).to_string())
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Normalize (log-normalize or DESeq2-style size factor correction)")
    print("  2. Filter lowly expressed genes (keep genes expressed in 2+ samples)")
    print("  3. Run transpose.py on the output")
    print("  4. Run file_splitter.py for Weka feature selection")


# ---------------------------------------------------------------------------
# MAIN
# Parses command-line arguments and runs the full pseudobulk pipeline.
# ---------------------------------------------------------------------------
DEFAULT_SAMPLES_DIR = "/data/femoral_head_necrosis_old"
DEFAULT_OUTPUT_CSV  = "/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/pseudobulk/pseudobulk_matrix.csv"

if __name__ == "__main__":
    if len(sys.argv) == 1:
        samples_dir = DEFAULT_SAMPLES_DIR
        output_csv  = DEFAULT_OUTPUT_CSV
    elif len(sys.argv) == 3:
        samples_dir = sys.argv[1]
        output_csv  = sys.argv[2]
    else:
        print("Usage: python3 pseudobulk.py [<samples_dir> <output_csv>]")
        print(f"  Default samples_dir : {DEFAULT_SAMPLES_DIR}")
        print(f"  Default output_csv  : {DEFAULT_OUTPUT_CSV}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Build the pseudobulk matrix from all 5 sample folders
    df = build_pseudobulk_matrix(samples_dir)

    # Print a human-readable summary before saving
    print_summary(df)

    # Save to CSV — this is the input file for all downstream steps
    df.to_csv(output_csv)
    print(f"\nSaved to: {output_csv}")
