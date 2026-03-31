import argparse
import pathlib
import re
import pandas as pd


def _find_best_match(canonical_symbol: str, gene_alias: str, df: pd.DataFrame):
    """
    Return the best-matching row for a known gene.
    Matching is done against gene_symbol using whole-token regex matching.
    """
    gene_col = df["gene_symbol"].fillna("").astype(str)

    def _mask_for(symbol: str):
        return gene_col.str.contains(
            rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
            regex=True,
            case=False,
            na=False,
        )

    hits = df[_mask_for(canonical_symbol)].copy()
    if hits.empty and gene_alias != canonical_symbol:
        hits = df[_mask_for(gene_alias)].copy()

    if hits.empty:
        return None

    # Sort by the best score columns that exist
    sort_cols = []
    ascending = []

    if "combined_score" in hits.columns:
        sort_cols.append("combined_score")
        ascending.append(False)
    if "Median_TestAUC" in hits.columns:
        sort_cols.append("Median_TestAUC")
        ascending.append(False)
    if "rf_importance" in hits.columns:
        sort_cols.append("rf_importance")
        ascending.append(False)
    if "selection_freq" in hits.columns:
        sort_cols.append("selection_freq")
        ascending.append(False)
    if "abs_fold_change" in hits.columns:
        sort_cols.append("abs_fold_change")
        ascending.append(False)

    if sort_cols:
        hits = hits.sort_values(by=sort_cols, ascending=ascending)

    return hits.iloc[0]


def _safe_get(row, col, ndigits=4):
    if col not in row.index:
        return ""
    val = row[col]
    if pd.isna(val):
        return ""
    try:
        return round(float(val), ndigits)
    except Exception:
        return val


def run_audit(shortlist_path: str, known_path: str, run_label: str, out_path: str) -> None:
    known = pd.read_csv(known_path)
    df = pd.read_csv(shortlist_path)

    required_cols = {"probe_id", "gene_symbol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    rows = []
    for _, ref in known.iterrows():
        gene = str(ref["gene"]).strip()
        canonical = str(ref["canonical_symbol"]).strip()

        best = _find_best_match(canonical, gene, df)

        if best is None:
            rows.append(
                {
                    "run_label": run_label,
                    "gene": gene,
                    "canonical_symbol": canonical,
                    "tier": ref.get("tier", ""),
                    "subcategory": ref.get("subcategory", ""),
                    "evidence_type": ref.get("evidence_type", ""),
                    "microarray_comparable": ref.get("microarray_comparable", ""),
                    "found": "N",
                    "probe_id": "",
                    "gene_symbol_in_run": "",
                    "rf_importance": "",
                    "selection_freq": "",
                    "abs_fold_change": "",
                    "combined_score": "",
                    "median_test_auc": "",
                    "sd_test_auc": "",
                    "notes": ref.get("notes", ""),
                }
            )
        else:
            rows.append(
                {
                    "run_label": run_label,
                    "gene": gene,
                    "canonical_symbol": canonical,
                    "tier": ref.get("tier", ""),
                    "subcategory": ref.get("subcategory", ""),
                    "evidence_type": ref.get("evidence_type", ""),
                    "microarray_comparable": ref.get("microarray_comparable", ""),
                    "found": "Y",
                    "probe_id": best.get("probe_id", ""),
                    "gene_symbol_in_run": best.get("gene_symbol", ""),
                    "rf_importance": _safe_get(best, "rf_importance"),
                    "selection_freq": _safe_get(best, "selection_freq"),
                    "abs_fold_change": _safe_get(best, "abs_fold_change"),
                    "combined_score": _safe_get(best, "combined_score"),
                    "median_test_auc": _safe_get(best, "Median_TestAUC"),
                    "sd_test_auc": _safe_get(best, "SD_TestAUC"),
                    "notes": ref.get("notes", ""),
                }
            )

    audit = pd.DataFrame(rows)

    # Sort with whatever score columns exist in the audit
    sort_cols = ["tier", "found"]
    ascending = [True, False]

    if "combined_score" in audit.columns:
        sort_cols.append("combined_score")
        ascending.append(False)
    elif "median_test_auc" in audit.columns:
        sort_cols.append("median_test_auc")
        ascending.append(False)

    audit = audit.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)

    out_path_obj = pathlib.Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_path_obj, index=False)

    found_count = int((audit["found"] == "Y").sum())
    missing_count = int((audit["found"] == "N").sum())

    print("\n" + "─" * 60)
    print("SONFH known-gene audit")
    print("─" * 60)
    print(f"Run label       : {run_label}")
    print(f"Input path      : {shortlist_path}")
    print(f"Known genes     : {known_path}")
    print(f"Rows in input   : {len(df)}")
    print(f"Known genes hit : {found_count} / {len(audit)}")
    print(f"Known genes miss: {missing_count}")
    print(f"Output written  : {out_path}")

    if found_count:
        print("\nTop recovered known genes:")
        recovered = audit[audit["found"] == "Y"].copy()

        if "combined_score" in recovered.columns and recovered["combined_score"].notna().any():
            recovered = recovered.sort_values(by=["combined_score"], ascending=[False])
            for _, row in recovered.head(15).iterrows():
                print(
                    f"  {row['canonical_symbol']:<12} "
                    f"{row['probe_id']:<18} "
                    f"score={row['combined_score']:<6} "
                    f"fc={row['abs_fold_change']}"
                )
        else:
            recovered = recovered.sort_values(by=["median_test_auc"], ascending=[False])
            for _, row in recovered.head(15).iterrows():
                print(
                    f"  {row['canonical_symbol']:<12} "
                    f"{row['probe_id']:<18} "
                    f"auc={row['median_test_auc']:<6} "
                    f"fc={row['abs_fold_change']}"
                )

    if missing_count:
        print("\nMissing known genes:")
        missing_rows = audit[audit["found"] == "N"]
        for _, row in missing_rows.iterrows():
            print(f"  {row['canonical_symbol']:<12} tier={row['tier']}")

    print("─" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONFH known-gene audit")
    parser.add_argument(
        "--shortlist",
        required=True,
        help="Path to biomarker_shortlist.csv, ann_probe_ranking.csv, or equivalent CSV",
    )
    parser.add_argument(
        "--known",
        default="report/sonfh_known_genes.csv",
        help="Path to known-gene master list CSV",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Run label, e.g. multivariate_top100_min_s_50",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output audit CSV path",
    )
    args = parser.parse_args()

    run_audit(
        shortlist_path=args.shortlist,
        known_path=args.known,
        run_label=args.label,
        out_path=args.out,
    )

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_multivariate_top100_min_s_50 / biomarker_shortlist.csv \
    #- -label
    #multivariate_top100_min_s_50 \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_multivariate_top100_min_s_50.csv

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_multivariate_top500_min_s_45 / biomarker_shortlist.csv \
    #- -label
    #multivariate_top500_min_s_45 \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_multivariate_top500_min_s_45.csv

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_univariate_top100_min_s_50 / biomarker_shortlist.csv \
    #- -label
    #univariate_top100_min_s_50 \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_univariate_top100_min_s_50.csv

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_univariate_top500_min_s_45 / biomarker_shortlist.csv \
    #- -label
    #univariate_top500_min_s_45 \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_univariate_top500_min_s_45.csv

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_univariate_top100_min_s_50 / feature_selection / univariate_ann / ann_probe_ranking.csv \
    #- -label
    #univariate_ann_ranking_top100_run \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_univariate_ann_ranking_top100_run.csv

    #python3
    #report / generate_sonfh_gene_audit.py \
    #- -shortlist / Users / jordanharris / Code / Omics_Capstone / omics_ml_pipeline / app / data / output_univariate_top500_min_s_45 / feature_selection / univariate_ann / ann_probe_ranking.csv \
    #- -label
    #univariate_ann_ranking_top500_min_s_45 \
    #- -out / Users / jordanharris / Code / Omics_Capstone / report / audit_univariate_ann_ranking_top500_min_s_45.csv