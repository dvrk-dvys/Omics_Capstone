"""
file_splitter.py — Python equivalent of 'file splitter.R'

What the R script does:
  For a wide CSV with structure:
    [ID_col | gene1 | gene2 | ... | geneN | class_col]
  It loops through each middle column (gene columns) and saves a separate CSV:
    [ID_col | geneX | class_col]
  Each file is named:  01_gene1.csv, 02_gene2.csv, etc.

  This feeds one gene at a time into the single-gene ANN wrapper.
  NOTE: This is NOT for Weka directly — Weka takes the full transposed matrix.

Usage:
  python3 file_splitter.py <input.csv> <output_dir>
  python3 file_splitter.py "data_transposed.csv" "split_output"
"""

import sys
import re
import os
import pandas as pd


def split_by_gene(input_path: str, output_dir: str) -> None:
    df = pd.read_csv(input_path, dtype=str)

    os.makedirs(output_dir, exist_ok=True)

    first_col = df.columns[0]
    last_col  = df.columns[-1]
    mid_cols  = df.columns[1:-1]  # all gene columns

    print(f"Input shape   : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"ID column     : '{first_col}'")
    print(f"Class column  : '{last_col}'")
    print(f"Gene columns  : {len(mid_cols)} (will produce {len(mid_cols)} files)\n")

    for i, col in enumerate(mid_cols, start=1):
        subset = df[[first_col, col, last_col]]

        # Sanitise column name for filename (match R's gsub("[^A-Za-z0-9_]", "_", ...))
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", col)
        filename  = f"{i:02d}_{safe_name}.csv"
        out_path  = os.path.join(output_dir, filename)

        subset.to_csv(out_path, index=False)
        print(f"Saved: {filename}")

    print(f"\nAll {len(mid_cols)} files saved to: {output_dir}/")

    # --- Weka readiness note ---
    print("\n--- Note on Weka use ---")
    print("These split files are designed for the single-gene ANN wrapper (one gene at a time).")
    print("For Weka classifiers, use the FULL transposed matrix (transpose.py output) instead.")
    print(f"Each split file has 3 columns: [{first_col}, <gene>, {last_col}]")
    print(f"Class column '{last_col}' is last — correct position for Weka.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 file_splitter.py <input.csv> <output_dir>")
        sys.exit(1)
    split_by_gene(sys.argv[1], sys.argv[2])
