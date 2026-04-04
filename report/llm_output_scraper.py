import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── RUN LABELS ───────────────────────────────────────────────────────────────
# Labels are derived dynamically from JSON filename prefixes (everything before
# the first '_evidence' segment).  No manual mapping needed — add new runs by
# dropping a new JSON into best_runs/ and re-running with --analyze-all.


# ─── ORIGINAL SCRAPER FUNCTIONS (single-run mode, unchanged) ─────────────────

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


# ─── ANALYZE-ALL: DATA LOADING ────────────────────────────────────────────────

def load_known_genes(folder: Path) -> List[Dict]:
    """
    Load all CSV files from `folder` as reference gene sets for known/validated genes.

    Required column (at least one must be present per CSV):
        'gene'             — primary gene symbol column
        'canonical_symbol' — accepted fallback if 'gene' is absent

    Optional columns stored and used in table/txt outputs:
        tier               — integer evidence tier (e.g. 1, 2) for sonfh-style lists
        subcategory        — biological category or pathway label
        stage              — study stage string for validation lists (e.g. 'early', 'late')
        rtqpcr_validated   — 'Y' / 'N' RT-qPCR validation flag

    All other columns are silently dropped (not stored, not written to any output).

    Returns a list of source dicts sorted alphabetically by filename — the first
    entry becomes KnownA in reports, second becomes KnownB, etc.:
        [{"source": "<filename_stem>", "key_col": "<col>", "data": {gene: {meta}}}, ...]

    Control A/B assignment order by renaming files (alphabetical ordering applies).
    """
    KEEP_COLS = {"tier", "subcategory", "stage", "rtqpcr_validated"}
    sources = []

    for csv_path in sorted(folder.glob("*.csv")):
        rows: Dict[str, Dict] = {}
        key_col: Optional[str] = None
        has_tier = False

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            has_tier = "tier" in reader.fieldnames
            if "gene" in reader.fieldnames:
                key_col = "gene"
            elif "canonical_symbol" in reader.fieldnames:
                key_col = "canonical_symbol"
            else:
                print(f"[SKIP] {csv_path.name}: no 'gene' or 'canonical_symbol' column")
                continue

            for row in reader:
                gene = row.get(key_col, "").strip()
                if not gene:
                    continue
                kept = {k: v for k, v in row.items() if k in KEEP_COLS and v}
                rows[gene] = kept

        if rows:
            sources.append({"source": csv_path.stem, "key_col": key_col, "data": rows, "has_tier": has_tier})
            print(f"[KNOWN] {csv_path.name} → {len(rows)} genes loaded")

    # Sort: tier-bearing files (sonfh-style) → KnownA first; stage/validation files → KnownB
    sources.sort(key=lambda s: (0 if s.get("has_tier") else 1, s["source"]))

    for i, s in enumerate(sources):
        print(f"[KNOWN] Known{chr(ord('A') + i)} = {s['source']}")

    return sources


def _label_from_filename(stem: str) -> str:
    """Extract run label from JSON filename stem: the prefix before the first '_evidence' segment."""
    idx = stem.find("_evidence")
    return stem[:idx] if idx != -1 else stem


def load_all_runs(best_runs_dir: Path):
    """
    Load all JSON files from best_runs_dir.
    Returns (runs, run_filters) where:
        runs        = {run_label: [records]}
        run_filters = {run_label: filter_str}  e.g. 'evidence_tier_Tier_1_OR_...'
    """
    runs: Dict[str, List[Dict]] = {}
    run_filters: Dict[str, str] = {}
    for json_path in sorted(best_runs_dir.glob("*.json")):
        label = _label_from_filename(json_path.stem)
        filter_str = json_path.stem[len(label) + 1:] if json_path.stem.startswith(label + "_") else ""
        try:
            with open(json_path, encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list):
                runs[label] = records
                run_filters[label] = filter_str
                print(f"[LOAD] {json_path.name} → {label} ({len(records)} records)")
            else:
                print(f"[SKIP] {json_path.name}: expected a list at root")
        except Exception as e:
            print(f"[SKIP] {json_path.name}: {e}")
    return runs, run_filters


# ─── ANALYZE-ALL: FORMATTING HELPERS ─────────────────────────────────────────

def _tier_short(tier: str) -> str:
    """'Tier 1' → 'T1',  empty/None → '—'"""
    return tier.replace("Tier ", "T") if tier else "—"


def _rel_short(rel: str) -> str:
    """'direct' → 'd',  'indirect' → 'i'"""
    if rel == "direct":
        return "d"
    if rel == "indirect":
        return "i"
    return "?"


def _known_a_display(meta: Dict) -> str:
    """Render KnownA cell: 'T{tier}' if tier present, else '—'."""
    tier = meta.get("tier", "")
    return f"T{tier}" if tier else "—"


def _known_b_display(meta: Dict) -> str:
    """Render KnownB cell: stage if present, else rtqpcr_validated, else '—'."""
    return meta.get("stage") or meta.get("rtqpcr_validated") or "—"


def _write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _build_gene_run_map(runs: Dict[str, List[Dict]], run_order: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Build a unified gene → {run_label → record_fields} map used across all tables."""
    gene_run_map: Dict[str, Dict[str, Dict]] = {}
    for run in run_order:
        for rec in runs.get(run, []):
            gene = rec.get("gene_symbol", "")
            if not gene:
                continue
            gene_run_map.setdefault(gene, {})[run] = {
                "tier":       rec.get("evidence_tier", ""),
                "score":      rec.get("score"),
                "rel":        rec.get("evidence_relation", ""),
                "fc":         rec.get("abs_fold_change"),
                "confidence": rec.get("evidence_confidence", ""),
                "bio_pot":    rec.get("biomarker_potential", ""),
            }
    return gene_run_map


def _gene_sort_key(gene: str, gene_run_map: Dict) -> tuple:
    """Sort key: n_runs desc, then best score desc."""
    rd = gene_run_map[gene]
    scores = [v["score"] for v in rd.values() if v["score"] is not None]
    return (-len(rd), -(max(scores) if scores else 0))


# ─── ANALYZE-ALL: CSV TABLES ──────────────────────────────────────────────────

def build_csv_tables(
    runs: Dict[str, List[Dict]],
    known_sources: List[Dict],
    tables_dir: Path,
    run_filters: Dict[str, str] = None,
) -> None:
    """Write one CSV per run plus summary tables 2–5 to tables_dir."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    if run_filters is None:
        run_filters = {}

    run_order    = sorted(runs.keys())
    known_a      = known_sources[0]["data"] if len(known_sources) > 0 else {}
    known_b      = known_sources[1]["data"] if len(known_sources) > 1 else {}
    known_a_name = known_sources[0]["source"] if len(known_sources) > 0 else "known_a"
    known_b_name = known_sources[1]["source"] if len(known_sources) > 1 else "known_b"

    gene_run_map = _build_gene_run_map(runs, run_order)
    all_genes = sorted(gene_run_map.keys(), key=lambda g: _gene_sort_key(g, gene_run_map))

    # ── Per-run detail tables (one CSV per run) ──────────────────────────────
    for run in run_order:
        rows_run = []
        for rank, rec in enumerate(runs[run], 1):
            gene = rec.get("gene_symbol", "")
            ka   = known_a.get(gene, {})
            kb   = known_b.get(gene, {})
            rows_run.append({
                "rank":                          rank,
                "gene":                          gene,
                "tier":                          rec.get("evidence_tier", ""),
                "relation":                      rec.get("evidence_relation", ""),
                "confidence":                    rec.get("evidence_confidence", ""),
                "biomarker_potential":           rec.get("biomarker_potential", ""),
                "score":                         rec.get("score", ""),
                "abs_fold_change":               rec.get("abs_fold_change", ""),
                f"known_{known_a_name}":          ka.get("tier", ""),
                f"known_{known_b_name}":          kb.get("stage", "") or kb.get("rtqpcr_validated", ""),
            })
        _write_csv(tables_dir / f"{run}.csv", rows_run)

    # ── Table 2: aligned gene matrix ────────────────────────────────────────
    rows_t2 = []
    for gene in all_genes:
        run_data = gene_run_map[gene]
        fcs = [v["fc"] for v in run_data.values() if v["fc"] is not None]
        row: Dict[str, Any] = {"gene": gene}
        for run in run_order:
            r = run_data.get(run, {})
            row[f"{run}_tier"]  = r.get("tier", "")
            row[f"{run}_score"] = r.get("score", "")
            row[f"{run}_rel"]   = r.get("rel", "")
        row["best_fc"]                       = round(max(fcs), 3) if fcs else ""
        row["n_runs"]                        = len(run_data)
        row[f"known_{known_a_name}"]         = known_a.get(gene, {}).get("tier", "")
        row[f"known_{known_b_name}"]         = known_b.get(gene, {}).get("stage", "") or known_b.get(gene, {}).get("rtqpcr_validated", "")
        rows_t2.append(row)

    cols_t2 = (
        ["gene"]
        + [f"{run}_{s}" for run in run_order for s in ("tier", "score", "rel")]
        + ["best_fc", "n_runs", f"known_{known_a_name}", f"known_{known_b_name}"]
    )
    _write_csv(tables_dir / "table2_aligned_gene_matrix.csv", rows_t2, cols_t2)

    # ── Table 3: tier/relation counts per run ────────────────────────────────
    rows_t3 = []
    for run in run_order:
        c = {k: 0 for k in ["T1_direct", "T1_indirect", "T2_direct", "T2_indirect", "T3_direct", "T3_indirect"]}
        ka_hits = kb_hits = 0
        scores = []
        tier_scores: Dict[str, List[float]] = {"T1": [], "T2": [], "T3": []}
        for rec in runs[run]:
            gene = rec.get("gene_symbol", "")
            tier = rec.get("evidence_tier", "").replace("Tier ", "T")
            rel  = rec.get("evidence_relation", "")
            key  = f"{tier}_{rel}"
            if key in c:
                c[key] += 1
            if gene in known_a:
                ka_hits += 1
            if gene in known_b:
                kb_hits += 1
            s = rec.get("score")
            if s is not None:
                scores.append(s)
                if tier in tier_scores:
                    tier_scores[tier].append(s)
        avg_score    = round(sum(scores) / len(scores), 4) if scores else ""
        avg_score_t1 = round(sum(tier_scores["T1"]) / len(tier_scores["T1"]), 4) if tier_scores["T1"] else ""
        avg_score_t2 = round(sum(tier_scores["T2"]) / len(tier_scores["T2"]), 4) if tier_scores["T2"] else ""
        avg_score_t3 = round(sum(tier_scores["T3"]) / len(tier_scores["T3"]), 4) if tier_scores["T3"] else ""
        rows_t3.append({
            "run": run, **c, "total": sum(c.values()), "avg_score": avg_score,
            f"{known_a_name}_hits": ka_hits, f"{known_b_name}_hits": kb_hits,
            "avg_score_T1": avg_score_t1, "avg_score_T2": avg_score_t2, "avg_score_T3": avg_score_t3,
        })

    # Union row: deduplicated counts across all pipelines
    union_c: Dict[str, set] = {k: set() for k in ["T1_direct", "T1_indirect", "T2_direct", "T2_indirect", "T3_direct", "T3_indirect"]}
    union_ka: set = set()
    union_kb: set = set()
    gene_best_score_t3: Dict[str, float] = {}
    union_tier_genes: Dict[str, Dict[str, float]] = {"T1": {}, "T2": {}, "T3": {}}

    for gene, run_data in gene_run_map.items():
        for rv in run_data.values():
            tier = rv["tier"].replace("Tier ", "T") if rv["tier"] else ""
            key  = f"{tier}_{rv['rel']}"
            if key in union_c:
                union_c[key].add(gene)
            s = rv["score"]
            if s is not None:
                gene_best_score_t3[gene] = max(gene_best_score_t3.get(gene, s), s)
                if tier in union_tier_genes:
                    prev = union_tier_genes[tier].get(gene)
                    union_tier_genes[tier][gene] = max(prev, s) if prev is not None else s
        if gene in known_a:
            union_ka.add(gene)
        if gene in known_b:
            union_kb.add(gene)

    all_best = list(gene_best_score_t3.values())
    union_avg = round(sum(all_best) / len(all_best), 4) if all_best else ""
    union_t_avgs: Dict[str, Any] = {}
    for t in ("T1", "T2", "T3"):
        s_list = list(union_tier_genes[t].values())
        union_t_avgs[t] = round(sum(s_list) / len(s_list), 4) if s_list else ""

    rows_t3.append({
        "run": "union_all_pipelines",
        **{k: len(v) for k, v in union_c.items()},
        "total": len(gene_run_map),
        "avg_score": union_avg,
        f"{known_a_name}_hits": len(union_ka),
        f"{known_b_name}_hits": len(union_kb),
        "avg_score_T1": union_t_avgs["T1"],
        "avg_score_T2": union_t_avgs["T2"],
        "avg_score_T3": union_t_avgs["T3"],
    })
    _write_csv(tables_dir / "table3_tier_relation_counts.csv", rows_t3)

    # ── Table 4: cross-pipeline convergence (2+ runs) ────────────────────────
    rows_t4 = []
    for gene in all_genes:
        run_data = gene_run_map[gene]
        if len(run_data) < 2:
            continue
        pipelines = "  ".join(r for r in run_order if r in run_data)
        tiers = [run_data[r]["tier"] for r in run_order if r in run_data and run_data[r]["tier"]]
        best_tier = min(tiers, key=lambda t: int(t.replace("Tier ", "")), default="") if tiers else ""
        fcs = [run_data[r]["fc"] for r in run_order if r in run_data and run_data[r]["fc"] is not None]
        all_direct = all(run_data[r].get("rel") == "direct" for r in run_order if r in run_data)
        rows_t4.append({
            "gene":       gene,
            "n_runs":     len(run_data),
            "pipelines":  pipelines,
            "best_tier":  best_tier.replace("Tier ", "T") if best_tier else "",
            "best_fc":    round(max(fcs), 3) if fcs else "",
            "all_direct": all_direct,
            f"known_{known_a_name}": known_a.get(gene, {}).get("tier", ""),
            f"known_{known_b_name}": known_b.get(gene, {}).get("stage", "") or known_b.get(gene, {}).get("rtqpcr_validated", ""),
        })
    _write_csv(tables_dir / "table4_cross_pipeline_convergence.csv", rows_t4)

    # ── Table 5: known gene hits ─────────────────────────────────────────────
    rows_t5 = []
    for source_info in known_sources:
        src_label = source_info["source"]
        for gene in all_genes:
            meta = source_info["data"].get(gene)
            if meta is None:
                continue
            run_data = gene_run_map[gene]
            run_parts, best_score, best_fc_val, best_tier = [], None, None, None
            for r in run_order:
                if r not in run_data:
                    continue
                rd  = run_data[r]
                t   = _tier_short(rd["tier"])
                rel = _rel_short(rd["rel"])
                run_parts.append(f"{r}:{t}({rel})")
                s = rd.get("score")
                if s is not None and (best_score is None or s > best_score):
                    best_score = s
                fc = rd.get("fc")
                if fc is not None and (best_fc_val is None or fc > best_fc_val):
                    best_fc_val = fc
                if best_tier is None or (rd["tier"] and int(rd["tier"].replace("Tier ", "")) < int(best_tier.replace("Tier ", ""))):
                    best_tier = rd["tier"]
            rows_t5.append({
                "gene":            gene,
                "source":          src_label,
                "known_tier":      meta.get("tier", ""),
                "category":        meta.get("subcategory", ""),
                "runs_found":      "  ".join(run_parts),
                "best_tier_found": _tier_short(best_tier) if best_tier else "",
                "best_score":      round(best_score, 4) if best_score is not None else "",
                "best_fc":         round(best_fc_val, 3) if best_fc_val is not None else "",
                "jia2023_stage":   meta.get("stage", ""),
                "jia2023_rtqpcr":  meta.get("rtqpcr_validated", ""),
            })
    _write_csv(tables_dir / "table5_known_gene_hits.csv", rows_t5)

    # ── Table 6: per-pipeline evidence + signal summary ──────────────────────
    gene_best_fc_csv: Dict[str, Optional[float]] = {}
    for g in all_genes:
        fcs = [v["fc"] for v in gene_run_map[g].values() if v["fc"] is not None]
        gene_best_fc_csv[g] = max(fcs) if fcs else None

    rows_t6 = []
    for run in run_order:
        recs = runs[run]
        t1d = t1i = t2d = t2i = 0
        fc_t1: List[float] = []; fc_t2: List[float] = []
        fc_dir: List[float] = []; fc_ind: List[float] = []
        in_multi = 0
        for rec in recs:
            gene = rec.get("gene_symbol", "")
            tier = rec.get("evidence_tier", "")
            rel  = rec.get("evidence_relation", "")
            fc   = rec.get("abs_fold_change")
            if tier == "Tier 1" and rel == "direct":
                t1d += 1
                if fc is not None: fc_t1.append(fc); fc_dir.append(fc)
            elif tier == "Tier 1" and rel == "indirect":
                t1i += 1
                if fc is not None: fc_t1.append(fc); fc_ind.append(fc)
            elif tier == "Tier 2" and rel == "direct":
                t2d += 1
                if fc is not None: fc_t2.append(fc); fc_dir.append(fc)
            elif tier == "Tier 2" and rel == "indirect":
                t2i += 1
                if fc is not None: fc_t2.append(fc); fc_ind.append(fc)
            if gene in gene_run_map and len(gene_run_map[gene]) >= 2:
                in_multi += 1
        rows_t6.append({
            "pipeline":     run,
            "T1_direct":    t1d, "T1_indirect": t1i,
            "T2_direct":    t2d, "T2_indirect": t2i,
            "total_direct": t1d + t2d,
            "total_indirect": t1i + t2i,
            "total_genes":  len(recs),
            "in_2plus_pipelines": in_multi,
            "avg_fc_T1":    round(sum(fc_t1)/len(fc_t1), 3) if fc_t1 else "",
            "avg_fc_T2":    round(sum(fc_t2)/len(fc_t2), 3) if fc_t2 else "",
            "avg_fc_direct":   round(sum(fc_dir)/len(fc_dir), 3) if fc_dir else "",
            "avg_fc_indirect": round(sum(fc_ind)/len(fc_ind), 3) if fc_ind else "",
        })

    # Union row for CSV
    u_t1d: set = set(); u_t1i: set = set()
    u_t2d: set = set(); u_t2i: set = set()
    u_dir: set = set(); u_ind: set = set()
    for g in all_genes:
        for rv in gene_run_map[g].values():
            tier = rv["tier"]; rel = rv["rel"]
            if tier == "Tier 1" and rel == "direct":   u_t1d.add(g); u_dir.add(g)
            if tier == "Tier 1" and rel == "indirect":  u_t1i.add(g); u_ind.add(g)
            if tier == "Tier 2" and rel == "direct":   u_t2d.add(g); u_dir.add(g)
            if tier == "Tier 2" and rel == "indirect":  u_t2i.add(g); u_ind.add(g)

    def _avg_fc_set_csv(s):
        vals = [gene_best_fc_csv[g] for g in s if gene_best_fc_csv.get(g) is not None]
        return round(sum(vals)/len(vals), 3) if vals else ""

    u_multi = {g for g in all_genes if len(gene_run_map[g]) >= 2}
    rows_t6.append({
        "pipeline":     "Union — all pipelines",
        "T1_direct":    len(u_t1d), "T1_indirect": len(u_t1i),
        "T2_direct":    len(u_t2d), "T2_indirect": len(u_t2i),
        "total_direct": len(u_dir), "total_indirect": len(u_ind),
        "total_genes":  len(all_genes),
        "in_2plus_pipelines": len(u_multi),
        "avg_fc_T1":    _avg_fc_set_csv(u_t1d | u_t1i),
        "avg_fc_T2":    _avg_fc_set_csv(u_t2d | u_t2i),
        "avg_fc_direct":   _avg_fc_set_csv(u_dir),
        "avg_fc_indirect": _avg_fc_set_csv(u_ind),
    })
    _write_csv(tables_dir / "table6_pipeline_signal_summary.csv", rows_t6)

    print(f"[TABLES] Wrote {len(run_order)} per-run CSVs + 5 summary tables → {tables_dir}")


# ─── ANALYZE-ALL: TXT REPORT ─────────────────────────────────────────────────

_SEP = "=" * 108
_DIV = "─" * 90


def build_txt_report(
    runs: Dict[str, List[Dict]],
    known_sources: List[Dict],
    output_path: Path,
    run_filters: Dict[str, str] = None,
) -> None:
    """Write the full human-readable analysis report to output_path."""
    if run_filters is None:
        run_filters = {}

    run_order    = sorted(runs.keys())
    known_a      = known_sources[0]["data"] if len(known_sources) > 0 else {}
    known_b      = known_sources[1]["data"] if len(known_sources) > 1 else {}
    known_a_name = known_sources[0]["source"] if len(known_sources) > 0 else "known_a"
    known_b_name = known_sources[1]["source"] if len(known_sources) > 1 else ""

    gene_run_map = _build_gene_run_map(runs, run_order)
    all_genes    = sorted(gene_run_map.keys(), key=lambda g: _gene_sort_key(g, gene_run_map))

    # Dynamic widths shared across tables
    run_col_w = max((len(r) for r in run_order), default=5) + 2
    RCW       = max((len(r) for r in run_order), default=6) + 2  # matrix column width

    lines = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    unique_filters = sorted(set(run_filters.values()))
    known_label_parts = [
        f"Known {chr(ord('A') + i)} = {ks['source']}.csv"
        for i, ks in enumerate(known_sources)
    ]

    lines.append(_SEP)
    lines.append(f"LLM OUTPUT ANALYSIS — ALL {len(run_order)} PIPELINE RUNS")
    if len(unique_filters) == 1:
        lines.append(f"Filters applied: {unique_filters[0]}")
    elif unique_filters:
        for fstr in unique_filters:
            lines.append(f"Filter: {fstr}")
    lines.append("  ".join(known_label_parts))
    lines.append(_SEP)
    lines.append("")

    # ── PER-RUN TABLES ────────────────────────────────────────────────────────
    HDR = (
        f"{'#':<4}{'Gene':<11}{'Tier':<6}{'Relation':<11}"
        f"{'Conf':<11}{'Bio.Pot':<11}{'Score':<9}{'FC':<8}{'KnownA':<9}KnownB"
    )

    for run in run_order:
        recs = runs[run]
        t_counts = {(t, r): 0 for t in ("Tier 1", "Tier 2", "Tier 3") for r in ("direct", "indirect")}
        for rec in recs:
            key = (rec.get("evidence_tier", ""), rec.get("evidence_relation", ""))
            if key in t_counts:
                t_counts[key] += 1

        t1d = t_counts[("Tier 1", "direct")];  t1i = t_counts[("Tier 1", "indirect")]
        t2d = t_counts[("Tier 2", "direct")];  t2i = t_counts[("Tier 2", "indirect")]
        t3d = t_counts[("Tier 3", "direct")];  t3i = t_counts[("Tier 3", "indirect")]
        t3_part = (
            f"  |  T3: {t3d+t3i} (direct={t3d}, indirect={t3i})"
            if (t3d + t3i) > 0 else "  |  T3: 0"
        )

        lines.append(f"┌── {run}")
        lines.append(
            f"│   Total: {len(recs)}"
            f"  |  T1: {t1d+t1i} (direct={t1d}, indirect={t1i})"
            f"  |  T2: {t2d+t2i} (direct={t2d}, indirect={t2i})"
            f"{t3_part}"
        )
        filt = run_filters.get(run, "")
        if filt:
            lines.append(f"│   Filter: {filt}")
        lines.append(f"└{'─' * 90}")
        lines.append(HDR)
        lines.append("─" * 90)

        for rank, rec in enumerate(recs, 1):
            gene  = rec.get("gene_symbol", "")
            tier  = _tier_short(rec.get("evidence_tier", ""))
            rel   = rec.get("evidence_relation", "")
            conf  = rec.get("evidence_confidence", "")
            bio   = rec.get("biomarker_potential", "")
            score = rec.get("score")
            fc    = rec.get("abs_fold_change")
            sc_s  = f"{score:.4f}" if score is not None else "—"
            fc_s  = f"{fc:.3f}"   if fc    is not None else "—"
            ka    = _known_a_display(known_a.get(gene, {}))
            kb    = _known_b_display(known_b.get(gene, {}))
            lines.append(
                f"{rank:<4}{gene:<11}{tier:<6}{rel:<11}{conf:<11}{bio:<11}{sc_s:<9}{fc_s:<8}{ka:<9}{kb}"
            )

        lines.append("")

    # ── TABLE 1: ALIGNED GENE MATRIX ────────────────────────────────────────
    lines.append(_SEP)
    lines.append("TABLE 1 — ALIGNED GENE MATRIX  (d=direct  i=indirect)")
    lines.append(_SEP)

    mat_hdr = (
        f"{'Gene':<12}"
        + "".join(f"{r:<{RCW}}" for r in run_order)
        + f"{'FC':<7}{'Runs':<6}{'KnownA':<8}KnownB"
    )
    mat_div = "─" * (12 + len(run_order) * RCW + 21)
    lines.append(mat_hdr)
    lines.append(mat_div)

    for gene in all_genes:
        run_data = gene_run_map[gene]
        fcs  = [v["fc"] for v in run_data.values() if v["fc"] is not None]
        cells = []
        for run in run_order:
            if run in run_data:
                rd   = run_data[run]
                t    = _tier_short(rd["tier"])
                sc   = f"{rd['score']:.2f}" if rd["score"] is not None else "?"
                r    = _rel_short(rd["rel"])
                cell = f"{t}({sc}){r}"
            else:
                cell = "—"
            cells.append(f"{cell:<{RCW}}")
        fc_s = f"{max(fcs):.3f}" if fcs else "—"
        ka   = _known_a_display(known_a.get(gene, {}))
        kb   = _known_b_display(known_b.get(gene, {}))
        lines.append(
            f"{gene:<12}" + "".join(cells) + f"{fc_s:<7}{len(run_data):<6}{ka:<8}{kb}"
        )

    lines.append("")

    # ── TABLE 2: TIER & RELATION COUNTS PER RUN ──────────────────────────────
    div2_len = run_col_w + 5 + 5 + 5 + 5 + 5 + 6 + 8 + 8 + 6
    lines.append(_SEP)
    lines.append("TABLE 2 — TIER & RELATION COUNTS PER RUN")
    lines.append(_SEP)
    lines.append(
        f"{'Run':<{run_col_w}}{'T1d':<5}{'T1i':<5}{'T2d':<5}{'T2i':<5}{'T3d':<5}{'T3i':<6}{'Total':<8}{'KnownA':<8}KnownB"
    )
    lines.append("─" * div2_len)

    for run in run_order:
        recs = runs[run]
        c = {k: 0 for k in ["T1d", "T1i", "T2d", "T2i", "T3d", "T3i"]}
        ka_hits = kb_hits = 0
        for rec in recs:
            gene = rec.get("gene_symbol", "")
            t    = rec.get("evidence_tier", "").replace("Tier ", "T")
            r    = "d" if rec.get("evidence_relation") == "direct" else "i"
            key  = f"{t}{r}"
            if key in c:
                c[key] += 1
            if gene in known_a:
                ka_hits += 1
            if gene in known_b:
                kb_hits += 1
        total = sum(c.values())
        lines.append(
            f"{run:<{run_col_w}}{c['T1d']:<5}{c['T1i']:<5}{c['T2d']:<5}{c['T2i']:<5}"
            f"{c['T3d']:<5}{c['T3i']:<6}{total:<8}{ka_hits:<8}{kb_hits}"
        )

    lines.append("")

    # ── TABLE 3: CROSS-PIPELINE CONVERGENCE (2+ runs) ────────────────────────
    convergent_genes = [g for g in all_genes if len(gene_run_map[g]) >= 2]
    max_pipe_len = max(
        (len("  ".join(r for r in run_order if r in gene_run_map[g])) for g in convergent_genes),
        default=36,
    )
    pipe_col_w = max(max_pipe_len + 2, len("Pipelines") + 2)
    div3_len = 12 + 6 + pipe_col_w + 13 + 9 + 8 + 8

    lines.append(_SEP)
    lines.append("TABLE 3 — CROSS-PIPELINE CONVERGENCE (found in 2+ runs)")
    lines.append(_SEP)
    lines.append(
        f"{'Gene':<12}{'Runs':<6}{'Pipelines':<{pipe_col_w}}{'Best Tier':<13}{'Best FC':<9}{'KnownA':<8}KnownB"
    )
    lines.append("─" * div3_len)

    for gene in all_genes:
        run_data = gene_run_map[gene]
        if len(run_data) < 2:
            continue
        pipelines = "  ".join(r for r in run_order if r in run_data)
        tiers = [run_data[r]["tier"] for r in run_order if r in run_data and run_data[r]["tier"]]
        best_tier = min(tiers, key=lambda t: int(t.replace("Tier ", "")), default="") if tiers else ""
        fcs  = [run_data[r]["fc"] for r in run_order if r in run_data and run_data[r]["fc"] is not None]
        t_s  = best_tier.replace("Tier ", "T") if best_tier else "—"
        fc_s = f"{max(fcs):.3f}" if fcs else "—"
        ka   = _known_a_display(known_a.get(gene, {}))
        kb   = _known_b_display(known_b.get(gene, {}))
        lines.append(
            f"{gene:<12}{len(run_data):<6}{pipelines:<{pipe_col_w}}{t_s:<13}{fc_s:<9}{ka:<8}{kb}"
        )

    lines.append("")

    # ── KNOWN GENE TABLES (one per source) ───────────────────────────────────
    table_num = 4
    for source_info in known_sources:
        src_name = source_info["source"]
        src_data = source_info["data"]
        has_tier = source_info.get("has_tier", False)

        # Pre-compute max runs string length for this source's matched genes
        max_runs_len = max(
            (
                len("  ".join(
                    f"{r}:{_tier_short(gene_run_map[g][r]['tier'])}({gene_run_map[g][r]['score']:.2f}){_rel_short(gene_run_map[g][r]['rel'])}"
                    for r in run_order if r in gene_run_map.get(g, {})
                ))
                for g in all_genes if g in src_data and g in gene_run_map
            ),
            default=len("Runs"),
        )
        runs_col_w = max(max_runs_len + 2, len("Runs") + 2)

        lines.append(_SEP)
        if has_tier:
            lines.append(f"TABLE {table_num} — KNOWN GENE HITS ({src_name}.csv)")
            lines.append(_SEP)
            lines.append(f"{'Gene':<11}{'KnownT':<9}{'Category':<26}{'Runs':<{runs_col_w}}Best FC")
            lines.append("─" * (11 + 9 + 26 + runs_col_w + 7))
        else:
            lines.append(f"TABLE {table_num} — {src_name} GENE HITS ({src_name}.csv)")
            lines.append(_SEP)
            lines.append(f"{'Gene':<11}{'Stage':<9}{'RT-qPCR':<13}{'Runs':<{runs_col_w}}Best FC")
            lines.append("─" * (11 + 9 + 13 + runs_col_w + 7))

        found_any = False
        for gene in all_genes:
            meta = src_data.get(gene)
            if meta is None:
                continue
            run_data = gene_run_map.get(gene, {})
            run_parts, best_fc_val = [], None
            for r in run_order:
                if r not in run_data:
                    continue
                rd  = run_data[r]
                t   = _tier_short(rd["tier"])
                sc  = f"{rd['score']:.2f}" if rd["score"] is not None else "?"
                rel = _rel_short(rd["rel"])
                run_parts.append(f"{r}:{t}({sc}){rel}")
                fc = rd.get("fc")
                if fc is not None and (best_fc_val is None or fc > best_fc_val):
                    best_fc_val = fc

            runs_str = "  ".join(run_parts)
            fc_s     = f"{best_fc_val:.3f}" if best_fc_val is not None else "—"

            if has_tier:
                kt  = f"T{meta.get('tier', '?')}" if meta.get("tier") else "—"
                cat = meta.get("subcategory", "—")
                lines.append(f"{gene:<11}{kt:<9}{cat:<26}{runs_str:<{runs_col_w}}{fc_s}")
            else:
                stage     = meta.get("stage", "—")
                validated = meta.get("rtqpcr_validated", "—")
                lines.append(f"{gene:<11}{stage:<9}{validated:<13}{runs_str:<{runs_col_w}}{fc_s}")
            found_any = True

        if not found_any:
            lines.append("(no matches found in this run set)")
        lines.append("")
        table_num += 1

    # ── TABLE 6: PER-PIPELINE EVIDENCE + SIGNAL SUMMARY ──────────────────────
    # Precompute best FC per gene (max across all runs)
    gene_best_fc: Dict[str, Optional[float]] = {}
    for g in all_genes:
        fcs = [v["fc"] for v in gene_run_map[g].values() if v["fc"] is not None]
        gene_best_fc[g] = max(fcs) if fcs else None

    def _avg_fc(gene_set) -> str:
        vals = [gene_best_fc[g] for g in gene_set if gene_best_fc.get(g) is not None]
        return f"{sum(vals) / len(vals):.3f}" if vals else "—"

    # Per-run rows
    t6_rows = []
    for run in run_order:
        recs = runs[run]
        t1d = t1i = t2d = t2i = 0
        fc_t1: List[float] = []
        fc_t2: List[float] = []
        fc_dir: List[float] = []
        fc_ind: List[float] = []
        in_multi = 0

        for rec in recs:
            gene = rec.get("gene_symbol", "")
            tier = rec.get("evidence_tier", "")
            rel  = rec.get("evidence_relation", "")
            fc   = rec.get("abs_fold_change")

            if tier == "Tier 1" and rel == "direct":
                t1d += 1
                if fc is not None:
                    fc_t1.append(fc); fc_dir.append(fc)
            elif tier == "Tier 1" and rel == "indirect":
                t1i += 1
                if fc is not None:
                    fc_t1.append(fc); fc_ind.append(fc)
            elif tier == "Tier 2" and rel == "direct":
                t2d += 1
                if fc is not None:
                    fc_t2.append(fc); fc_dir.append(fc)
            elif tier == "Tier 2" and rel == "indirect":
                t2i += 1
                if fc is not None:
                    fc_t2.append(fc); fc_ind.append(fc)

            if gene in gene_run_map and len(gene_run_map[gene]) >= 2:
                in_multi += 1

        t6_rows.append({
            "pipeline":    run,
            "T1d":         t1d,
            "T1i":         t1i,
            "T2d":         t2d,
            "T2i":         t2i,
            "tot_d":       t1d + t2d,
            "tot_i":       t1i + t2i,
            "total":       len(recs),
            "in_multi":    in_multi,
            "avg_fc_t1":   f"{sum(fc_t1)/len(fc_t1):.3f}" if fc_t1 else "—",
            "avg_fc_t2":   f"{sum(fc_t2)/len(fc_t2):.3f}" if fc_t2 else "—",
            "avg_fc_dir":  f"{sum(fc_dir)/len(fc_dir):.3f}" if fc_dir else "—",
            "avg_fc_ind":  f"{sum(fc_ind)/len(fc_ind):.3f}" if fc_ind else "—",
        })

    # Union row
    union_t1d: set = set(); union_t1i: set = set()
    union_t2d: set = set(); union_t2i: set = set()
    union_dir: set = set(); union_ind: set = set()
    for g in all_genes:
        for rv in gene_run_map[g].values():
            tier = rv["tier"]; rel = rv["rel"]
            if tier == "Tier 1" and rel == "direct":
                union_t1d.add(g); union_dir.add(g)
            if tier == "Tier 1" and rel == "indirect":
                union_t1i.add(g); union_ind.add(g)
            if tier == "Tier 2" and rel == "direct":
                union_t2d.add(g); union_dir.add(g)
            if tier == "Tier 2" and rel == "indirect":
                union_t2i.add(g); union_ind.add(g)
    union_t1 = union_t1d | union_t1i
    union_t2 = union_t2d | union_t2i
    union_multi = {g for g in all_genes if len(gene_run_map[g]) >= 2}

    t6_rows.append({
        "pipeline":    "Union — all pipelines",
        "T1d":         len(union_t1d),
        "T1i":         len(union_t1i),
        "T2d":         len(union_t2d),
        "T2i":         len(union_t2i),
        "tot_d":       len(union_dir),
        "tot_i":       len(union_ind),
        "total":       len(all_genes),
        "in_multi":    len(union_multi),
        "avg_fc_t1":   _avg_fc(union_t1),
        "avg_fc_t2":   _avg_fc(union_t2),
        "avg_fc_dir":  _avg_fc(union_dir),
        "avg_fc_ind":  _avg_fc(union_ind),
    })

    # Render TABLE 6
    pipe_col_t6 = max(
        max((len(r) for r in run_order), default=5),
        len("Union — all pipelines"),
    ) + 2
    div6_len = pipe_col_t6 + 5 + 5 + 5 + 5 + 7 + 7 + 7 + 7 + 12 + 12 + 13 + 10

    lines.append(_SEP)
    lines.append(f"TABLE {table_num} — PER-PIPELINE EVIDENCE + SIGNAL SUMMARY")
    lines.append(
        "Per-pipeline counts of Tier 1 / Tier 2 evidence by relation type, "
        "plus average absolute fold-change (|FC|) for key evidence subsets."
    )
    lines.append(_SEP)
    lines.append(
        f"{'Pipeline':<{pipe_col_t6}}"
        f"{'T1d':<5}{'T1i':<5}{'T2d':<5}{'T2i':<5}"
        f"{'Tot.d':<7}{'Tot.i':<7}{'Total':<7}{'≥2pip':<7}"
        f"{'Avg|FC|T1':<12}{'Avg|FC|T2':<12}{'Avg|FC|dir':<13}Avg|FC|ind"
    )
    lines.append("─" * div6_len)

    for i, row in enumerate(t6_rows):
        if i == len(t6_rows) - 1:          # blank line before union row
            lines.append("")
        lines.append(
            f"{row['pipeline']:<{pipe_col_t6}}"
            f"{row['T1d']:<5}{row['T1i']:<5}{row['T2d']:<5}{row['T2i']:<5}"
            f"{row['tot_d']:<7}{row['tot_i']:<7}{row['total']:<7}{row['in_multi']:<7}"
            f"{row['avg_fc_t1']:<12}{row['avg_fc_t2']:<12}{row['avg_fc_dir']:<13}{row['avg_fc_ind']}"
        )

    lines.append("")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    lines.append(_SEP)
    lines.append("SUMMARY")
    lines.append(_SEP)

    all_gene_set  = set(gene_run_map.keys())
    all_known_set: set = set()
    lines.append(f"Total unique genes across all {len(run_order)} runs: {len(all_gene_set)}")

    for source_info in known_sources:
        matched = [g for g in all_genes if g in source_info["data"]]
        gene_list = ", ".join(matched)
        lines.append(f"Matched {source_info['source']}.csv: {len(matched):>3}  → {gene_list}")
        all_known_set |= set(source_info["data"].keys())

    novel      = [g for g in all_genes if g not in all_known_set]
    convergent = [g for g in all_genes if len(gene_run_map[g]) >= 2]
    lines.append(f"Novel (neither list):          {len(novel):>3}")
    lines.append(f"Cross-pipeline convergence 2+: {len(convergent):>3}")
    lines.append("")

    for run in run_order:
        recs = runs[run]
        t1 = sum(1 for r in recs if r.get("evidence_tier") == "Tier 1")
        t2 = sum(1 for r in recs if r.get("evidence_tier") == "Tier 2")
        t3 = sum(1 for r in recs if r.get("evidence_tier") == "Tier 3")
        d  = sum(1 for r in recs if r.get("evidence_relation") == "direct")
        ii = sum(1 for r in recs if r.get("evidence_relation") == "indirect")
        lines.append(
            f"{run:<{run_col_w}}:  {len(recs):>2} genes  |  T1={t1}  T2={t2}  T3={t3}"
            f"  |  direct={d}  indirect={ii}"
        )
    lines.append(_SEP)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TXT] Wrote report → {output_path}")


# ─── ANALYZE-ALL ENTRY POINT ──────────────────────────────────────────────────

def analyze_all_runs(args) -> None:
    best_runs_dir   = Path(args.input)
    known_genes_dir = Path(args.known_genes)
    out_dir         = Path(args.out_dir)
    tables_dir      = out_dir / "llm_analysis_tables"
    txt_path        = out_dir / "llm_output_analysis_all_runs.txt"

    print(f"[ANALYZE-ALL] best_runs dir  : {best_runs_dir}")
    print(f"[ANALYZE-ALL] known_genes dir: {known_genes_dir}")
    print(f"[ANALYZE-ALL] output dir     : {out_dir}")

    known_sources        = load_known_genes(known_genes_dir)
    runs, run_filters    = load_all_runs(best_runs_dir)

    if not runs:
        print("[ERROR] No run JSONs found in best_runs dir. Exiting.")
        return

    build_csv_tables(runs, known_sources, tables_dir, run_filters)
    build_txt_report(runs, known_sources, txt_path, run_filters)


# ─── ARG PARSING & MAIN ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Scrape and filter LLM gene interpretation outputs")

    # ── Shared args (used by both modes) ─────────────────────────────────────
    p.add_argument(
        "--out-dir",
        default="/Users/jordanharris/Code/Omics_Capstone/report",
        help="Root output directory (both modes)",
    )

    # ── Single-run scraper mode ───────────────────────────────────────────────
    p.add_argument(
        "--input",
        #default="/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/llm_outputs/multivariate",            # weka_multivariate
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_weka_multivariate/llm_outputs",  # weka_multivariate
        default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_weka_univariate_ann/llm_outputs",  # weka_univariate_ann
        #default="/Users/jordanharris/Code/Omics_Capstone/data/femoral_head_necrosis/llm_outputs/univariate_ann",           # weka_univariate_ann
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_multivariate_top500/llm_outputs",   # new_multivariate_top500
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_multivariate_top100/llm_outputs",   # new_multivariate_top100
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_new_univariate_top100/llm_outputs",     # new_univariate_top100
        #default="/Users/jordanharris/Code/Omics_Capstone/AutoOmics_ML_Pipeline/app/data/output_univariate_rerank_top100/llm_outputs",  # univariate_rerank_top100
        #default="/Users/jordanharris/Code/Omics_Capstone/report/best_runs",  # ← use this for --analyze-all
        help="llm_outputs dir (single-run mode) OR best_runs JSON dir (--analyze-all)",
    )
    p.add_argument(
        "--tag",
        default="weka_univariate_ann",
        help="Output filename prefix for single-run mode (e.g. 'weka_multi', 'weka_uni')",
    )

    # ── Analyze-all mode ─────────────────────────────────────────────────────
    p.add_argument(
        "--analyze-all",
        action="store_true",
        help=(
            "Read all JSONs from --input (best_runs dir), generate CSV tables "
            "and txt report. Switch --input to the report/best_runs default above."
        ),
    )
    p.add_argument(
        "--known-genes",
        default="/Users/jordanharris/Code/Omics_Capstone/report/known_genes",
        help=(
            "Folder of reference CSV files for known/validated genes (--analyze-all only). "
            "Required column per CSV: 'gene' or 'canonical_symbol'. "
            "CSVs are loaded in alphabetical order: first → KnownA, second → KnownB, etc. "
            "Rename files to control ordering."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.analyze_all:
        analyze_all_runs(args)
        return

    # ── Single-run scraper (original behavior) ────────────────────────────────
    input_dir = Path(args.input)

    # DEFINE YOUR FILTERS HERE
    # Use a list for OR logic on the same field, single value for exact match
    filters = {
        "evidence_tier": ["Tier 1", "Tier 2", "Tier 3"],
        "evidence_relation": ["direct", "indirect"],  # "direct",
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
