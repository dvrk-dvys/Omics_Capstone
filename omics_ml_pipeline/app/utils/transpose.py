"""
transpose.py — Python equivalent of Tranpose_Function.R

What the R script does:
  1. read.csv(file, colClasses="character")  — reads everything as strings
  2. t(data)                                  — transposes (genes->rows becomes genes->cols)
  3. write.table(result, sep=",", ...)        — writes back as CSV

Usage:
  python3 transpose.py <input.csv> <output.csv>
  python3 transpose.py "data for courseweka.csv" "data_transposed.csv"
"""

import sys
import pandas as pd


def transpose_csv(input_path: str, output_path: str) -> None:
    # Read as strings (matches R's colClasses="character")
    df = pd.read_csv(input_path, index_col=0, dtype=str)

    print(f"Input shape : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Input preview (top-left):\n{df.iloc[:3, :4]}\n")

    df_t = df.T

    print(f"Transposed shape: {df_t.shape[0]} rows x {df_t.shape[1]} cols")
    print(f"Transposed preview (top-left):\n{df_t.iloc[:3, :4]}\n")

    df_t.to_csv(output_path)
    print(f"Saved to: {output_path}")

    # --- Weka readiness check ---
    print("\n--- Weka Readiness Check ---")
    print(f"Rows (samples)  : {df_t.shape[0]}")
    print(f"Cols (features) : {df_t.shape[1]}")
    print(f"Last column     : '{df_t.columns[-1]}'  <- should be your class label")
    print(f"Missing values  : {df_t.isnull().sum().sum()}")
    print("Format: rows=samples, cols=features+class  ->  ready for Weka .arff or .csv import")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 transpose.py <input.csv> <output.csv>")
        sys.exit(1)
    transpose_csv(sys.argv[1], sys.argv[2])
