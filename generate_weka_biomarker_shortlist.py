"""
generate_weka_biomarker_shortlist.py
─────────────────────────────────────
Parses Weka classifier output files (RF + optional J48) and produces a
biomarker shortlist CSV in the same style as the Python pipeline output.

Background
──────────
Two Weka models produce actionable gene candidates — all others (NaiveBayes,
SMO, MLP, IBk) are evaluation-only and produce accuracy numbers, not gene lists:

  J48   → split-node probes: one probe per decision boundary, directly citable
  RF    → attribute importance ranking: top-ranked probes across 200 trees

The shortlist = RF top features + J48 split nodes (when J48 output is provided),
annotated with gene symbols and fold-change from the Python pipeline rankings.

Usage
─────
# Multivariate path (Weka ran on top100_features.arff)
python3 generate_weka_biomarker_shortlist.py \\
    --rf    data/femoral_head_necrosis/weka_models/randomforest.txt \\
    --j48   data/femoral_head_necrosis/weka_models_old/j48_tree.txt \\
    --ranks data/femoral_head_necrosis/feature_selection/gene_rankings.csv \\
    --out   data/femoral_head_necrosis/weka_biomarker_shortlist.csv

# Univariate ANN path (Weka ran on top100_features_univariate_ann.arff)
python3 generate_weka_biomarker_shortlist.py \\
    --rf    data/weka/univariate/randomforest.txt \\
    --j48   data/weka/univariate/j48_tree.txt \\
    --ranks omics_ml_pipeline/app/data/output/feature_selection/gene_rankings.csv \\
    --out   data/weka/univariate/weka_biomarker_shortlist.csv

Arguments
─────────
  --rf              Path to Weka RandomForest output .txt  (required)
  --j48             Path to Weka J48 output .txt           (optional)
  --ranks           Path to gene_rankings.csv for symbol + fold-change annotation
                    (optional — probes listed without annotation if omitted)
  --out             Output CSV path                        (default: weka_biomarker_shortlist.csv)
  --min-importance  RF importance threshold (0–1, default 0.0 — include all)
  --top-n           Keep only the top N probes by RF importance (default: all)
"""

import argparse
import re
import pathlib
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------

def parse_rf(path: str) -> pd.DataFrame:
    """
    Parse Weka RandomForest attribute importance output.

    Looks for lines in the section:
      "Attribute importance based on average impurity decrease ..."
    with format:
      <score>  (<n_nodes>)  <probe_id>

    Returns DataFrame with columns: probe_id, rf_importance, rf_n_nodes
    Sorted descending by rf_importance.
    """
    pattern = re.compile(
        r"^\s+([\d.]+)\s+\(\s*(\d+)\s*\)\s+(\S+)\s*$"
    )
    rows = []
    in_importance = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Attribute importance based on" in line:
                in_importance = True
                continue
            if in_importance:
                m = pattern.match(line)
                if m:
                    importance = float(m.group(1))
                    n_nodes    = int(m.group(2))
                    probe_id   = m.group(3)
                    rows.append({
                        "probe_id":      probe_id,
                        "rf_importance": importance,
                        "rf_n_nodes":    n_nodes,
                    })
                elif rows and line.strip() == "":
                    # blank line after importance block — section ended
                    break
                elif rows and not pattern.match(line) and line.strip():
                    # non-matching non-blank line after block started — section ended
                    break

    if not rows:
        raise ValueError(
            f"No attribute importance data found in {path}.\n"
            "Make sure the file was saved from a Weka RF run with "
            "-attribute-importance enabled."
        )

    df = pd.DataFrame(rows).sort_values("rf_importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rf_rank", df.index + 1)
    return df


def parse_j48(path: str) -> pd.DataFrame:
    """
    Parse Weka J48 decision tree output and extract split-node probe IDs.

    Looks for lines in the tree section between:
      "J48 pruned tree" (or "J48 unpruned tree") and "Number of Leaves"
    with format:
      [indent] probe_id <= threshold: class (N/M)
      [indent] probe_id > threshold: class (N)

    Returns DataFrame with columns: probe_id, j48_threshold, j48_direction, j48_depth
    """
    split_pattern = re.compile(
        r"^(\s*)(\S+)\s+([<>]=?)\s+([\d.]+)\s*:"
    )
    rows = []
    in_tree = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip()
            if re.search(r"J48 (pruned|unpruned) tree", stripped):
                in_tree = True
                continue
            if in_tree and "Number of Leaves" in stripped:
                break
            if in_tree:
                m = split_pattern.match(stripped)
                if m:
                    depth     = len(m.group(1)) // 4    # Weka indents 4 spaces per level
                    probe_id  = m.group(2)
                    direction = m.group(3)
                    threshold = float(m.group(4))
                    rows.append({
                        "probe_id":       probe_id,
                        "j48_threshold":  threshold,
                        "j48_direction":  direction,
                        "j48_depth":      depth,
                    })

    if not rows:
        raise ValueError(
            f"No J48 split nodes found in {path}.\n"
            "Make sure the file was saved from a Weka J48 run."
        )

    # Deduplicate — same probe may appear at multiple branches
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["probe_id"])
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# ANNOTATION
# ---------------------------------------------------------------------------

def load_rankings(path: str) -> pd.DataFrame:
    """
    Load gene_rankings.csv (probe_id index, gene_symbol + fold-change columns).
    Returns a DataFrame indexed by probe_id.
    """
    df = pd.read_csv(path, index_col=0)
    df.index.name = "probe_id"
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_shortlist(
    rf_path,
    j48_path,
    rankings_path,
    out_path,
    min_importance,
    top_n,
):

    print(f"\n{'─' * 60}")
    print(f"Weka Biomarker Shortlist Generator")
    print(f"{'─' * 60}")

    # --- RF ---
    print(f"\nParsing RF importance: {rf_path}")
    rf_df = parse_rf(rf_path)
    print(f"  Found {len(rf_df)} probes in RF importance table")

    # Apply filters
    if min_importance > 0:
        rf_df = rf_df[rf_df["rf_importance"] >= min_importance].copy()
        print(f"  After min_importance >= {min_importance}: {len(rf_df)} probes")
    if top_n is not None:
        rf_df = rf_df.head(top_n).copy()
        print(f"  After top-{top_n} filter: {len(rf_df)} probes")

    shortlist = rf_df.copy()
    shortlist["j48_split"]      = False
    shortlist["j48_threshold"]  = None
    shortlist["j48_direction"]  = None
    shortlist["j48_depth"]      = None
    shortlist["source"]         = "rf"

    # --- J48 (optional) ---
    if j48_path:
        print(f"\nParsing J48 tree: {j48_path}")
        try:
            j48_df = parse_j48(j48_path)
            print(f"  Found {len(j48_df)} unique split-node probe(s) in J48 tree")

            # Mark RF rows that also appear as J48 split nodes
            j48_probes = set(j48_df["probe_id"])
            shortlist.loc[shortlist["probe_id"].isin(j48_probes), "j48_split"] = True
            shortlist.loc[shortlist["probe_id"].isin(j48_probes), "source"]    = "rf+j48"

            # Add J48 annotation columns for probes already in shortlist
            j48_idx = j48_df.set_index("probe_id")
            for col in ["j48_threshold", "j48_direction", "j48_depth"]:
                shortlist[col] = shortlist["probe_id"].map(j48_idx[col])

            # Add any J48 probes NOT already in the RF list
            new_j48 = j48_df[~j48_df["probe_id"].isin(set(shortlist["probe_id"]))].copy()
            if not new_j48.empty:
                extra = pd.DataFrame({
                    "rf_rank":        None,
                    "probe_id":       new_j48["probe_id"].values,
                    "rf_importance":  None,
                    "rf_n_nodes":     None,
                    "j48_split":      True,
                    "j48_threshold":  new_j48["j48_threshold"].values,
                    "j48_direction":  new_j48["j48_direction"].values,
                    "j48_depth":      new_j48["j48_depth"].values,
                    "source":         "j48",
                })
                shortlist = pd.concat([shortlist, extra], ignore_index=True)
                print(f"  Added {len(new_j48)} J48-only probe(s) not in RF list")

        except ValueError as e:
            print(f"  WARNING: {e}")
            print("  Continuing without J48 data.")

    # --- Annotation ---
    shortlist["gene_symbol"] = "---"
    shortlist["abs_fc"]      = None
    shortlist["log_fc"]      = None
    shortlist["hybrid_score"] = None

    if rankings_path:
        print(f"\nAnnotating from gene rankings: {rankings_path}")
        try:
            rank_df = load_rankings(rankings_path)
            for col_out, col_in in [
                ("gene_symbol", "gene_symbol"),
                ("abs_fc",      "abs_fold_change"),
                ("log_fc",      "log_fold_change"),
                ("hybrid_score","hybrid_score"),
            ]:
                if col_in in rank_df.columns:
                    shortlist[col_out] = shortlist["probe_id"].map(rank_df[col_in])
            shortlist["gene_symbol"] = shortlist["gene_symbol"].fillna("---")
            print(f"  Annotated {(shortlist['gene_symbol'] != '---').sum()} / {len(shortlist)} probes")
        except Exception as e:
            print(f"  WARNING: Could not load rankings ({e}) — probes unannotated")

    # --- Final column order ---
    col_order = [
        "rf_rank", "probe_id", "gene_symbol",
        "rf_importance", "rf_n_nodes",
        "j48_split", "j48_threshold", "j48_direction", "j48_depth",
        "abs_fc", "log_fc", "hybrid_score",
        "source",
    ]
    shortlist = shortlist[[c for c in col_order if c in shortlist.columns]]

    # --- Write ---
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_path, index=False)

    # --- Summary ---
    n_rf_only = (shortlist["source"] == "rf").sum()
    n_both    = (shortlist["source"] == "rf+j48").sum()
    n_j48_only= (shortlist["source"] == "j48").sum() if "source" in shortlist.columns else 0

    print(f"\n{'─' * 60}")
    print(f"Shortlist written: {out_path}")
    print(f"  Total probes : {len(shortlist)}")
    print(f"  RF only      : {n_rf_only}")
    print(f"  RF + J48     : {n_both}")
    print(f"  J48 only     : {n_j48_only}")
    print(f"{'─' * 60}\n")

    if not shortlist.empty:
        print("Top entries:")
        preview_cols = ["probe_id", "gene_symbol", "rf_importance", "j48_split", "source"]
        preview_cols = [c for c in preview_cols if c in shortlist.columns]
        print(shortlist[preview_cols].head(15).to_string(index=False))

    return shortlist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Weka biomarker shortlist from RF + optional J48 output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rf",             required=True,  help="Weka RandomForest output .txt")
    p.add_argument("--j48",            default=None,   help="Weka J48 output .txt (optional)")
    p.add_argument("--ranks",          default=None,   help="gene_rankings.csv for annotation (optional)")
    p.add_argument("--out",            default="weka_biomarker_shortlist.csv", help="Output CSV path")
    p.add_argument("--min-importance", type=float, default=0.0,
                   help="Minimum RF importance to include (0–1, default 0.0)")
    p.add_argument("--top-n",          type=int,   default=None,
                   help="Keep only top-N probes by RF importance (default: all above threshold)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_shortlist(
        rf_path        = args.rf,
        j48_path       = args.j48,
        rankings_path  = args.ranks,
        out_path       = args.out,
        min_importance = args.min_importance,
        top_n          = args.top_n,
    )
