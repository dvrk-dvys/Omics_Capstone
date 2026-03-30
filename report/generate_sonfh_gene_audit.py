"""
generate_sonfh_gene_audit.py
────────────────────────────
Cross-references the static known-gene reference list against the current
pipeline outputs and writes a fresh sonfh_gene_audit.csv.

Reads:
  report/sonfh_known_genes.csv                                  ← static reference
  data/femoral_head_necrosis/feature_selection/gene_rankings.csv
  data/femoral_head_necrosis/feature_selection/top100_features.csv
  data/femoral_head_necrosis/feature_selection/top500_features.csv

Writes:
  report/sonfh_gene_audit.csv

Run from project root:
  python3 generate_sonfh_gene_audit.py
"""

import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_REPORT_DIR   = pathlib.Path(__file__).resolve().parent        # report/
_ROOT         = _REPORT_DIR.parent                              # project root
_FEAT_DIR     = _ROOT / "data" / "femoral_head_necrosis" / "feature_selection"

KNOWN_GENES   = str(_REPORT_DIR / "sonfh_known_genes.csv")
RANKINGS      = str(_FEAT_DIR   / "gene_rankings.csv")
TOP100        = str(_FEAT_DIR   / "top100_features.csv")
TOP500        = str(_FEAT_DIR   / "top500_features.csv")
AUDIT_OUT     = str(_REPORT_DIR / "sonfh_gene_audit.csv")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def best_probe_for_gene(gene: str, rankings: pd.DataFrame):
    """
    Return the highest-ranked (lowest probe_rank) row for a gene.
    Handles multi-gene probe annotations like 'FCGR2A /// FCGR2C' by checking
    whether the gene symbol appears anywhere in the annotation string.
    """
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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
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

        # Try canonical first, then alias
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

    # ── Summary ──────────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
