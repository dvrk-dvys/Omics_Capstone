import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def matches_filters(record: Dict, filters: Dict[str, Any]) -> bool:
    """
    Return True if record matches ALL filter conditions.
    List values are treated as OR (record must match any one of them).
    """
    for field, expected in filters.items():
        actual = record.get(field)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def collect_reports(input_dir: Path, filters: Dict[str, Any]) -> List[Dict]:
    """
    Scan input_dir for JSON reports matching all filters.
    """
    hits = []

    for path in input_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception as e:
            print(f"[SKIP] {path.name}: {e}")
            continue

        if matches_filters(record, filters):
            hits.append({
                "file": path.name,
                "probe_id": record.get("probe_id"),
                "gene_symbol": record.get("gene_symbol"),
                "evidence_tier": record.get("evidence_tier"),
                "evidence_relation": record.get("evidence_relation"),
                "evidence_confidence": record.get("evidence_confidence"),
                "biomarker_potential": record.get("biomarker_potential"),
                "score": record.get("score"),
                "abs_fold_change": record.get("abs_fold_change"),
                "relevance_summary": record.get("relevance_summary"),
                "citations": record.get("citations", []),
            })

    return hits


def write_json(output_path: Path, data: List[Dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_output_name(filters: Dict[str, Any]) -> str:
    parts = []
    for k, v in filters.items():
        if isinstance(v, list):
            value_str = "_OR_".join(str(x).replace(" ", "_") for x in v)
        else:
            value_str = str(v).replace(" ", "_")
        parts.append(f"{k}_{value_str}")
    return "_AND_".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="Scrape and filter LLM gene interpretation outputs")
    p.add_argument(
        "--input",
        #default="/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/llm_outputs/multivariate",           # weka_multivariate
        #default="/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/llm_outputs/univariate_ann",          # weka_univariate_ann
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_multivariate_top500/llm_outputs",        # new_multivariate_top500
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_multivariate_top100/llm_outputs",        # new_multivariate_top100
        default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_univariate_top100/llm_outputs",          # new_univariate_top100
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_univariate_rerank_top100/llm_outputs",       # univariate_rerank_top100
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output/llm_outputs",
        help="Path to llm_outputs directory",
    )
    p.add_argument(
        "--out-dir",
        default="/Users/jordanharris/Code/Omics_Capstone/report",
        help="Directory to write output JSON",
    )
    p.add_argument(
        "--tag",
        default="new_univariate_top100",
        help="Prefix for output filename to distinguish runs (e.g. 'weka_multi', 'weka_uni')",
    )
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input)

    # DEFINE YOUR FILTERS HERE
    # Use a list for OR logic on the same field, single value for exact match
    filters = {
        "evidence_tier": ["Tier 1", "Tier 2", "Tier 3"],
        "evidence_relation": ["direct", "indirect"],#"direct",
        # "evidence_confidence": "high",   # optional
        # "biomarker_potential": "strong"  # optional
    }

    filter_name = build_output_name(filters)
    output_name = f"{args.tag}_{filter_name}" if args.tag else filter_name
    output_file = Path(args.out_dir) / f"{output_name}.json"

    print(f"[START] Scanning: {input_dir}")
    print(f"[FILTERS] {filters}")

    hits = collect_reports(input_dir, filters)

    # Sort by best available score column
    sort_columns = [
        "score",             # Weka branches (normalized in llm_job)
        "combined_score",    # Python multivariate
        "univariate_score",  # Python univariate augmented
        "Median_TestAUC",    # Python univariate baseline
    ]

    sort_col = None
    for col in sort_columns:
        if hits and col in hits[0]:
            hits = sorted(
                hits,
                key=lambda x: (x.get(col) is not None, x.get(col)),
                reverse=True,
            )
            sort_col = col
            print(f"[SORT] Sorted by {col}")
            break

    if hits:
        for r in hits:
            print(f"  {r['gene_symbol']} | {sort_col}={r.get(sort_col)}")

    write_json(output_file, hits)
    print(f"[DONE] Wrote {len(hits)} reports → {output_file}")


if __name__ == "__main__":
    main()
