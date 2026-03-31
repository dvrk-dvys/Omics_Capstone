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
                "Median_TestAUC": record.get("Median_TestAUC"),
                "abs_fold_change": record.get("abs_fold_change"),
                "univariate_score": record.get("univariate_score"),
                "relevance_summary": record.get("relevance_summary"),
                "citations": record.get("citations", []),
            })

    return hits


def write_json(output_path: Path, data: List[Dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_output_name(filters: Dict[str, Any]) -> str:
    """
    Create a clean, readable filename from filters.
    """
    parts = []

    for k, v in filters.items():
        if isinstance(v, list):
            value_str = "_OR_".join(str(x).replace(" ", "_") for x in v)
        else:
            value_str = str(v).replace(" ", "_")

        parts.append(f"{k}_{value_str}")

    return "_AND_".join(parts)


def main():
    input_dir = Path(
        #"/app/data/output_univariate_rerank_top/llm_outputs"
                "/Users/jordanharris/Code/Omics_Capstone/omics_ml_pipeline/app/data/output_multivariate_top500_min_s_45/llm_outputs"
    )

    # DEFINE YOUR FILTERS HERE
    # Use a list for OR logic on the same field, single value for exact match
    filters = {
        "evidence_tier": ["Tier 1", "Tier 2", "Tier 3"],
        "evidence_relation": "direct",
        # "evidence_confidence": "high",   # optional
        # "biomarker_potential": "strong"  # optional
    }

    output_name = build_output_name(filters)
    output_file = Path(
        f"/Users/jordanharris/Code/Omics_Capstone/report/{output_name}.json"
    )

    print(f"[START] Scanning: {input_dir}")
    print(f"[FILTERS] {filters}")

    hits = collect_reports(input_dir, filters)

    # Sort by best available score column
    sort_columns = [
        "combined_score",    # multivariate
        "univariate_score",  # univariate augmented
        "Median_TestAUC",    # univariate baseline
    ]

    sort_col = None
    for col in sort_columns:
        if hits and col in hits[0]:
            hits = sorted(
                hits,
                key=lambda x: (x.get(col) is not None, x.get(col)),
                reverse=True
            )
            sort_col = col
            print(f"[SORT] Sorted by {col}")
            break

    if hits:
        #print("[TOP 5]")
        for r in hits:
            print(f"  {r['gene_symbol']} | {sort_col}={r.get(sort_col)}")

    write_json(output_file, hits)

    print(f"[DONE] Wrote {len(hits)} reports → {output_file}")


if __name__ == "__main__":
    main()
