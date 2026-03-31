"""
generate_sonfh_gene_audit.py
────────────────────────────
Cross-references the static known-gene reference list against pipeline outputs.

Usage:
  python3 report/generate_sonfh_gene_audit.py                  # multivariate (default)
  python3 report/generate_sonfh_gene_audit.py --mode univariate
  python3 report/generate_sonfh_gene_audit.py --mode multivariate

Multivariate mode reads (old path):
  report/sonfh_known_genes.csv
  data/femoral_head_necrosis/feature_selection/gene_rankings.csv
  data/femoral_head_necrosis/feature_selection/top100_features.csv
  data/femoral_head_necrosis/feature_selection/top500_features.csv
  → writes report/sonfh_gene_audit.csv

Univariate mode reads (new pipeline):
  report/sonfh_known_genes.csv
  omics_ml_pipeline/app/data/output/feature_selection/univariate_ann/biomarker_shortlist.csv
  omics_ml_pipeline/app/data/output/feature_selection/univariate_ann/top100_features_univariate_ann.csv
  omics_ml_pipeline/app/data/output/feature_selection/univariate_ann/top500_features_univariate_ann.csv
  → writes report/sonfh_gene_audit_univariate.csv
"""

import argparse
import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_REPORT_DIR  = pathlib.Path(__file__).resolve().parent        # report/
_ROOT        = _REPORT_DIR.parent                             # project root
_FEAT_DIR    = _ROOT / "data" / "femoral_head_necrosis" / "feature_selection"
_UNI_DIR     = (_ROOT / "omics_ml_pipeline" / "app" / "data" /
                "output" / "feature_selection" / "univariate_ann")

KNOWN_GENES  = str(_REPORT_DIR / "sonfh_known_genes.csv")

# multivariate
RANKINGS     = str(_FEAT_DIR / "gene_rankings.csv")
TOP100       = str(_FEAT_DIR / "top100_features.csv")
TOP500       = str(_FEAT_DIR / "top500_features.csv")
AUDIT_OUT    = str(_REPORT_DIR / "sonfh_gene_audit.csv")

# univariate
UNI_RANKINGS = str(_UNI_DIR / "ann_probe_ranking.csv")
UNI_TOP100   = str(_UNI_DIR / "top100_features_univariate_ann.csv")
UNI_TOP500   = str(_UNI_DIR / "top500_features_univariate_ann.csv")
UNI_AUDIT_OUT = str(_REPORT_DIR / "sonfh_gene_audit_univariate.csv")


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------
def _best_probe(gene: str, rankings: pd.DataFrame, gene_col: str):
    """Return the top-ranked row for a gene symbol (handles /// multi-gene annotations)."""
    mask = rankings[gene_col].str.contains(
        rf"(?<![A-Z0-9]){gene}(?![A-Z0-9])", regex=True, case=False, na=False
    )
    hits = rankings[mask]
    if hits.empty:
        return None
    return hits.iloc[0]   # caller must ensure rankings is pre-sorted


def _count_probes(gene: str, rankings: pd.DataFrame, gene_col: str) -> int:
    mask = rankings[gene_col].str.contains(
        rf"(?<![A-Z0-9]){gene}(?![A-Z0-9])", regex=True, case=False, na=False
    )
    return int(mask.sum())


# ---------------------------------------------------------------------------
# MULTIVARIATE AUDIT  (original logic, unchanged)
# ---------------------------------------------------------------------------
def best_probe_for_gene(gene: str, rankings: pd.DataFrame):
    mask = rankings["gene_symbol"].str.contains(
        rf"(?<![A-Z0-9]){gene}(?![A-Z0-9])", regex=True, case=False, na=False
    )
    hits = rankings[mask]
    if hits.empty:
        return None
    return hits.sort_values("probe_rank").iloc[0]


def count_probes_for_gene(gene: str, rankings: pd.DataFrame) -> int:
    mask = rankings["gene_symbol"].str.contains(
        rf"(?<![A-Z0-9]){gene}(?![A-Z0-9])", regex=True, case=False, na=False
    )
    return int(mask.sum())


def main():
    known   = pd.read_csv(KNOWN_GENES)
    rank_df = pd.read_csv(RANKINGS)

    top100_probes = set(pd.read_csv(TOP100, index_col=0, nrows=0).columns)
    top500_probes = set(pd.read_csv(TOP500, index_col=0, nrows=0).columns)

    total_probes_in_data = len(rank_df)
    print(f"gene_rankings   : {total_probes_in_data:,} probes")
    print(f"top100 probes   : {len(top100_probes)}")
    print(f"top500 probes   : {len(top500_probes)}")
    print(f"known genes     : {len(known)}\n")

    rows = []
    for _, ref in known.iterrows():
        gene      = ref["gene"]
        canonical = ref["canonical_symbol"]

        best = best_probe_for_gene(canonical, rank_df)
        if best is None:
            best = best_probe_for_gene(gene, rank_df)

        n_probes = count_probes_for_gene(canonical, rank_df)
        if n_probes == 0:
            n_probes = count_probes_for_gene(gene, rank_df)

        if best is not None:
            probe_id   = best["probe_id"]
            rank       = int(best["probe_rank"])
            hyb_score  = round(best["hybrid_score"], 4)
            abs_fc     = round(best["abs_fold_change"], 4)
            log_fc     = round(best["log_fold_change"], 4)
            p_val      = f"{best['p_value']:.3e}"
            in_top100  = "Y" if probe_id in top100_probes else "N"
            in_top500  = "Y" if probe_id in top500_probes else "N"
        else:
            probe_id  = ""
            rank      = ""
            hyb_score = ""
            abs_fc    = ""
            log_fc    = ""
            p_val     = ""
            in_top100 = "N"
            in_top500 = "N"

        rows.append({
            "gene":               gene,
            "canonical_symbol":   canonical,
            "tier":               ref["tier"],
            "subcategory":        ref["subcategory"],
            "evidence_type":      ref["evidence_type"],
            "microarray_comparable": ref["microarray_comparable"],
            "n_probes":           n_probes,
            "best_probe_id":      probe_id,
            "hybrid_score_rank":  rank,
            "hybrid_score":       hyb_score,
            "abs_fc":             abs_fc,
            "log_fc":             log_fc,
            "p_value":            p_val,
            "in_top100":          in_top100,
            "in_top500":          in_top500,
            "notes":              ref["notes"],
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_OUT, index=False)

    found   = audit["best_probe_id"].ne("").sum()
    missing = audit["best_probe_id"].eq("").sum()
    in100   = (audit["in_top100"] == "Y").sum()
    in500   = (audit["in_top500"] == "Y").sum()

    print(f"Probe found     : {found} / {len(audit)}")
    print(f"No probe        : {missing}")
    print(f"In top 100      : {in100}")
    print(f"In top 500      : {in500}")
    print(f"\nAudit written   : {AUDIT_OUT}")

    if missing:
        print("\nNo probe found for:")
        for _, r in audit[audit["best_probe_id"] == ""].iterrows():
            print(f"  {r['gene']:<14}  tier={r['tier']}  ({r['evidence_type']})")


# ---------------------------------------------------------------------------
# UNIVARIATE AUDIT  (new — checks survival through ANN 2000-probe filter)
# ---------------------------------------------------------------------------
def univariate_audit():
    """
    Check how many known SONFH biomarkers survived the univariate ANN pre-filter
    (2000-probe shortlist) and whether they reached top-100 / top-500.

    Rankings are already sorted by Median_TestAUC desc; rank = row index + 1.
    """
    known    = pd.read_csv(KNOWN_GENES)
    rank_df  = pd.read_csv(UNI_RANKINGS)   # probe_id, gene_symbol, Median_TestAUC, SD_TestAUC, ...
    rank_df  = rank_df.reset_index(drop=True)
    rank_df["ann_rank"] = rank_df.index + 1   # 1-based rank within the 2000

    # probe sets for top-100 / top-500 (first column = "sample", last = class label)
    top100_df    = pd.read_csv(UNI_TOP100, nrows=0)
    top500_df    = pd.read_csv(UNI_TOP500, nrows=0)
    top100_probes = set(c for c in top100_df.columns if c not in ("sample", "class_label", "class"))
    top500_probes = set(c for c in top500_df.columns if c not in ("sample", "class_label", "class"))

    print(f"ANN 2000-probe list : {len(rank_df):,} probes")
    print(f"top100 probes       : {len(top100_probes)}")
    print(f"top500 probes       : {len(top500_probes)}")
    print(f"known genes         : {len(known)}\n")

    rows = []
    for _, ref in known.iterrows():
        gene      = ref["gene"]
        canonical = ref["canonical_symbol"]

        best = _best_probe(canonical, rank_df, "gene_symbol")
        if best is None:
            best = _best_probe(gene, rank_df, "gene_symbol")

        n_probes = _count_probes(canonical, rank_df, "gene_symbol")
        if n_probes == 0:
            n_probes = _count_probes(gene, rank_df, "gene_symbol")

        if best is not None:
            probe_id   = best["probe_id"]
            ann_rank   = int(best["ann_rank"])
            med_auc    = round(float(best["Median_TestAUC"]), 4)
            sd_auc     = round(float(best["SD_TestAUC"]), 4) if pd.notna(best["SD_TestAUC"]) else ""
            in_2000    = "Y"
            in_top100  = "Y" if probe_id in top100_probes else "N"
            in_top500  = "Y" if probe_id in top500_probes else "N"
            if ann_rank <= 100:
                rank_bucket = "top100"
            elif ann_rank <= 500:
                rank_bucket = "top500"
            else:
                rank_bucket = "top2000"
            evidence_recovered = "Y" if in_top100 == "Y" else "N"
        else:
            probe_id           = ""
            ann_rank           = ""
            med_auc            = ""
            sd_auc             = ""
            in_2000            = "N"
            in_top100          = "N"
            in_top500          = "N"
            rank_bucket        = "not_found"
            evidence_recovered = "N"

        rows.append({
            "gene":               gene,
            "canonical_symbol":   canonical,
            "tier":               ref["tier"],
            "subcategory":        ref["subcategory"],
            "evidence_type":      ref["evidence_type"],
            "microarray_comparable": ref["microarray_comparable"],
            "n_probes_in_2000":   n_probes,
            "best_probe_id":      probe_id,
            "ann_rank":           ann_rank,
            "rank_bucket":        rank_bucket,
            "median_test_auc":    med_auc,
            "sd_test_auc":        sd_auc,
            "in_2000_filter":     in_2000,
            "in_top100":          in_top100,
            "in_top500":          in_top500,
            "evidence_recovered": evidence_recovered,
            "notes":              ref["notes"],
        })

    audit = pd.DataFrame(rows)
    audit = audit.sort_values(["tier", "ann_rank"], ascending=[True, True]).reset_index(drop=True)
    audit.to_csv(UNI_AUDIT_OUT, index=False)

    found    = audit["best_probe_id"].ne("").sum()
    missing  = audit["best_probe_id"].eq("").sum()
    in100    = (audit["in_top100"] == "Y").sum()
    in500    = (audit["in_top500"] == "Y").sum()
    tier1    = audit[audit["tier"].astype(str) == "1"]
    t1_found = tier1["best_probe_id"].ne("").sum()

    print(f"Survived 2000-filter : {found} / {len(audit)}  ({missing} not found)")
    print(f"  Tier-1 genes       : {t1_found} / {len(tier1)} survived")
    print(f"In top 100           : {in100}")
    print(f"In top 500           : {in500}")
    print(f"\nAudit written        : {UNI_AUDIT_OUT}")

    if missing:
        print("\nNOT in the 2000-probe filter:")
        for _, r in audit[audit["best_probe_id"] == ""].iterrows():
            print(f"  {r['gene']:<14}  tier={r['tier']}  ({r['evidence_type']})")

    print("\nTop results for known genes that survived:")
    survived = (audit[audit["best_probe_id"].ne("")]
                .sort_values("ann_rank"))
    for _, r in survived.iterrows():
        top_flag = ""
        if r["in_top100"] == "Y":
            top_flag = "  ← TOP100"
        elif r["in_top500"] == "Y":
            top_flag = "  ← top500"
        print(f"  rank {str(r['ann_rank']):>4}  AUC={r['median_test_auc']:.4f}  "
              f"{r['gene']:<14} ({r['best_probe_id']}){top_flag}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONFH gene audit")
    parser.add_argument(
        "--mode",
        choices=["multivariate", "univariate"],
        default="multivariate",
        help="Which pipeline output to audit (default: multivariate)",
    )
    args = parser.parse_args()

    if args.mode == "univariate":
        univariate_audit()
    else:
        main()
