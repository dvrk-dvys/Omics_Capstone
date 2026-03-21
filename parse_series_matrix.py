"""
parse_series_matrix.py — Extract data matrix from GEO series matrix file

WHY THIS SCRIPT EXISTS:
  Replaces pseudobulk.py for the GSE123568 microarray dataset.
  The old pipeline needed pseudobulk.py because scRNA-seq data had ~46,000
  individual cells that had to be summed into one row per patient sample.

  Microarray data is already at the sample level — one column per patient in
  the series matrix file. No cell aggregation is needed. This script just:
    1. Reads GSE123568_series_matrix.txt.gz
    2. Extracts sample metadata (which samples are SONFH vs control)
    3. Extracts the probe expression matrix (probes × samples)
    4. Transposes to (samples × probes) — the Weka orientation
    5. Appends a 'class' column (SONFH or control)
    6. Saves to CSV for preprocess.py

WHAT'S IN THE SERIES MATRIX FILE:
  Header lines (start with '!') — study and sample metadata
    !Sample_geo_accession  — GSM IDs in column order
    !Sample_title          — human-readable sample names (used for class detection)
    !series_matrix_table_begin — marks start of data
  Data block — tab-separated, probe_id as first column, one value per sample
    "ID_REF"  "GSM3507251"  "GSM3507252"  ...
    "probe_1"    7.23          7.45       ...
    "probe_2"    5.12          5.34       ...
    ...
    !series_matrix_table_end — marks end of data

DATA VALUES:
  Values are pre-processed log2 expression intensities (RMA normalization,
  standard for Affymetrix microarrays). They are already on a comparable scale
  across samples — no log-CPM transformation is needed. preprocess.py handles
  probe filtering (remove low-variance probes) but NOT re-normalization.

CLASS DETECTION:
  Based on the GEO page, sample titles contain "control group" or "SONFH group".
  This script detects class by looking for those strings (case-insensitive).
  If a sample title does not match either pattern, it is labelled 'unknown'
  and printed as a warning — you should inspect and relabel those rows manually.

OUTPUT:
  data/femoral_head_necrosis/parsed/parsed_matrix.csv
  Shape: 40 rows (samples) × (N probes + 1 class column)
  This feeds directly into preprocess.py.

Usage:
  python3 parse_series_matrix.py
  python3 parse_series_matrix.py <series_matrix.txt.gz> <output.csv>
"""

import sys
import os
import gzip
import io
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/GSE123568_series_matrix.txt.gz"
DEFAULT_OUTPUT = "/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/parsed/parsed_matrix.csv"


# ---------------------------------------------------------------------------
# PARSE HEADER
# ---------------------------------------------------------------------------
def parse_header(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    Read the '!' metadata lines to extract GSM accession IDs, sample titles,
    and the disease characteristic labels.

    For GSE123568, sample titles say "control group" / "disease group" — too vague.
    The characteristics line 'disease: SONFH' / 'disease: non-SONFH' is authoritative
    for class assignment and is what we actually use.

    Returns:
      gsm_ids      : list[str] — GSM accession IDs in column order
      titles       : list[str] — human-readable sample titles
      disease_vals : list[str] — values from the 'disease:' characteristics line
                                 (e.g. ['non-SONFH', ..., 'SONFH', ...])
    """
    gsm_ids      = []
    titles       = []
    disease_vals = []
    char_lines   = []   # all !Sample_characteristics_ch1 lines

    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                parts   = line.split("\t")
                gsm_ids = [p.strip('"') for p in parts[1:] if p.strip('"')]

            elif line.startswith("!Sample_title"):
                parts  = line.split("\t")
                titles = [p.strip('"') for p in parts[1:] if p.strip() != ""]

            elif line.startswith("!Sample_characteristics_ch1"):
                char_lines.append(line)

            elif line == "!series_matrix_table_begin":
                break

    # Find the characteristics line whose values start with "disease:"
    for char_line in char_lines:
        parts = char_line.split("\t")
        vals  = [p.strip('"') for p in parts[1:] if p.strip()]
        if vals and vals[0].startswith("disease:"):
            disease_vals = [v.replace("disease:", "").strip() for v in vals]
            break

    if not gsm_ids:
        raise ValueError("Could not find !Sample_geo_accession in the series matrix.")
    if not disease_vals:
        raise ValueError(
            "Could not find a 'disease:' characteristics line in the series matrix.\n"
            "Check the file manually with: gunzip -c <file> | grep Sample_characteristics"
        )

    print(f"Found {len(gsm_ids)} samples in header.")
    print(f"Disease values detected: {sorted(set(disease_vals))}")
    return gsm_ids, titles, disease_vals


# ---------------------------------------------------------------------------
# PARSE DATA MATRIX
# ---------------------------------------------------------------------------
def parse_data_matrix(path: str) -> pd.DataFrame:
    """
    Extract the expression data block between !series_matrix_table_begin
    and !series_matrix_table_end.

    The data block is tab-separated with probe IDs as the index column.
    Returns a DataFrame of shape (n_probes, n_samples).
    """
    data_lines = []
    in_data    = False

    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_data = True
                continue
            elif line == "!series_matrix_table_end":
                break
            elif in_data and line:
                data_lines.append(line)

    if not data_lines:
        raise ValueError(
            "No data found between !series_matrix_table_begin and !series_matrix_table_end.\n"
            "Check that the file is a valid GEO series matrix."
        )

    raw = "\n".join(data_lines)
    df  = pd.read_csv(io.StringIO(raw), sep="\t", index_col=0)

    # Strip any leftover quotes from probe IDs
    df.index = df.index.str.strip('"')
    # Strip quotes from column headers (GSM IDs)
    df.columns = df.columns.str.strip('"')

    print(f"Data matrix: {df.shape[0]} probes × {df.shape[1]} samples")
    return df


# ---------------------------------------------------------------------------
# ASSIGN CLASS LABELS
# ---------------------------------------------------------------------------
def assign_classes(gsm_ids: list[str], disease_vals: list[str]) -> dict[str, str]:
    """
    Map each GSM accession ID to a class label (SONFH or control).

    Uses the 'disease:' characteristics field, NOT the sample title.
    Titles for this dataset say 'control group' / 'disease group' which is ambiguous;
    the characteristics field 'disease: SONFH' / 'disease: non-SONFH' is authoritative.

    Detection logic (case-insensitive):
      'sonfh' but not 'non-sonfh'  →  'SONFH'
      'non-sonfh'                  →  'control'
      neither                      →  'unknown'  (warning printed)

    Parameters:
      gsm_ids      : list of GSM accession IDs
      disease_vals : list of disease characteristic values in matching order

    Returns:
      dict mapping GSM ID → class label string
    """
    class_map = {}
    for gsm, val in zip(gsm_ids, disease_vals):
        v = val.lower().strip()
        if v == "sonfh":
            class_map[gsm] = "SONFH"
        elif "non-sonfh" in v or v == "non_sonfh" or v == "control":
            class_map[gsm] = "control"
        else:
            class_map[gsm] = "unknown"
            print(f"  WARNING: unrecognised disease value '{val}' for {gsm}")

    counts = pd.Series(list(class_map.values())).value_counts()
    print(f"\nClass distribution:")
    for label, n in counts.items():
        print(f"  {label}: {n} samples")

    if "unknown" in class_map.values():
        print("\n  ACTION NEEDED: relabel 'unknown' samples above before running preprocess.py")

    return class_map


# ---------------------------------------------------------------------------
# BUILD FINAL MATRIX
# ---------------------------------------------------------------------------
def build_sample_matrix(data_df: pd.DataFrame, class_map: dict[str, str]) -> pd.DataFrame:
    """
    Transpose the data matrix from (probes × samples) to (samples × probes),
    then append the class column at the end.

    Parameters:
      data_df   : pd.DataFrame — probes × samples (from parse_data_matrix)
      class_map : dict          — GSM → class label (from assign_classes)

    Returns:
      pd.DataFrame — shape (n_samples, n_probes + 1), class column last
    """
    # Transpose: columns (GSM IDs) become rows
    df = data_df.T.copy()
    df.index.name = "sample"

    # Verify all column IDs have class assignments
    missing = [gsm for gsm in df.index if gsm not in class_map]
    if missing:
        raise ValueError(f"No class label for samples: {missing}")

    # Append class column last (Weka requirement)
    df["class"] = df.index.map(class_map)

    return df


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PARSED MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape         : {df.shape[0]} samples × {df.shape[1] - 1} probes (+class)")
    print(f"Class column  : '{df.columns[-1]}'  (last — correct for Weka)")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())
    print(f"\nSample index (first 5):")
    for name in df.index[:5]:
        print(f"  {name}  →  {df.loc[name, 'class']}")
    print(f"\nValue range (spot-check, first 10 probes):")
    vals = df.iloc[:, :10].values.flatten()
    print(f"  min={vals.min():.3f}  max={vals.max():.3f}  mean={vals.mean():.3f}")
    print("=" * 60)
    print("\nNext step:")
    print("  python3 preprocess.py")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_path  = DEFAULT_INPUT
        output_path = DEFAULT_OUTPUT
    elif len(sys.argv) == 3:
        input_path  = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print("Usage: python3 parse_series_matrix.py [<series_matrix.txt.gz> <output.csv>]")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("Download GSE123568_series_matrix.txt.gz from GEO and place it at that path.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}\n")

    # Step 1: Read header metadata
    gsm_ids, titles, disease_vals = parse_header(input_path)

    # Step 2: Read data matrix
    data_df = parse_data_matrix(input_path)

    # Step 3: Map class labels from 'disease:' characteristics field
    class_map = assign_classes(gsm_ids, disease_vals)

    # Step 4: Transpose + attach class
    df = build_sample_matrix(data_df, class_map)

    # Step 5: Summary
    print_summary(df)

    # Step 6: Save
    df.to_csv(output_path)
    print(f"\nSaved: {output_path}")
