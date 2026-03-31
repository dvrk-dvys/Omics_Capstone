"""
llm_job.py — LLM-assisted biological interpretation of biomarker candidates.

Reads the biomarker shortlist CSV, runs the 5-iteration agentic loop for each
unique gene, and writes one JSON file per gene to config["paths"]["llm_outputs_dir"].

Activated via:
  python -m app.main --llm
  python -m app.main --llm --shortlist path/to/custom_shortlist.csv
"""

import json
import os
import pathlib
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.llm.openai_client import DEFAULT_MODEL, agentic_llm
from app.utils.logging_utils import get_logger, console

log = get_logger("llm_job")


def _clean_gene(gene_symbol: str):
    """
    Return the primary gene token, or None to skip.
      'IGF2 /// INS-IGF2' → 'IGF2'
      '---'               → None
    """
    s = str(gene_symbol).strip()
    if s in ("---", "", "nan"):
        return None
    return s.split("///")[0].strip()


def _load_shortlist(path: str) -> list:
    """
    Load shortlist CSV, clean gene symbols, aggregate all probes per gene.

    For genes with multiple probes (e.g. TSTA3 appears 4×), keeps the best
    probe row as the base and adds aggregate columns so the saved JSON reflects
    the full multi-probe evidence:
      probe_count          — number of probes mapping to this gene
      probe_ids            — all probe IDs joined by " | "
      max_combined_score   — highest combined_score across all probes
      mean_abs_fold_change — mean absolute fold change across all probes
      max_rf_importance    — highest RF importance across all probes

    Returns list of row dicts sorted by max_combined_score descending.
    """
    df = pd.read_csv(path)
    df["gene_symbol"] = df["gene_symbol"].apply(_clean_gene)
    df = df.dropna(subset=["gene_symbol"])

    # Sort by whichever score column is present (multivariate uses combined_score;
    # univariate baseline uses Median_TestAUC; univariate augmented uses univariate_score)
    if "combined_score" in df.columns:
        _sort_col = "combined_score"
    elif "univariate_score" in df.columns:
        _sort_col = "univariate_score"
    else:
        _sort_col = "Median_TestAUC"
    df = df.sort_values(_sort_col, ascending=False)

    records = []
    for gene, group in df.groupby("gene_symbol", sort=False):
        best = group.iloc[0].to_dict()          # highest score row
        best["probe_count"]          = len(group)
        best["probe_ids"]            = " | ".join(group["probe_id"].tolist())
        best["max_combined_score"]   = round(float(group[_sort_col].max()), 4)
        best["mean_abs_fold_change"] = round(float(group["abs_fold_change"].mean()), 4)
        # rf_importance absent in univariate shortlists — default to 0
        best["max_rf_importance"]    = round(float(group["rf_importance"].max()), 4) if "rf_importance" in group.columns else 0.0
        records.append(best)

    records.sort(key=lambda r: r["max_combined_score"], reverse=True)
    return records


def _calc_cost(prompt_tokens: int, completion_tokens: int, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    return (prompt_tokens / 1_000_000) * input_cost_per_1m + (completion_tokens / 1_000_000) * output_cost_per_1m


def _process_gene(
    row: dict,
    out_dir: pathlib.Path,
    model: str,
    disease_context: str,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    openai_request_timeout: int = 300,
    tool_call_sleep: float = 0.5,
) -> dict:
    """Run agentic loop for one gene and return a status dict."""
    gene = row["gene_symbol"]
    probe = row.get("probe_id", "?")
    log.info(f"[START] gene={gene} probe={probe}")
    t0 = time.perf_counter()
    try:
        result, citations = agentic_llm(
            gene=gene,
            abstracts=[],
            model=model,
            temperature=0.1,
            max_tokens=500,
            disease_context=disease_context,
            openai_request_timeout=openai_request_timeout,
            tool_call_sleep=tool_call_sleep,
        )
    except Exception as agent_error:
        elapsed = time.perf_counter() - t0
        log.error(f"[FAIL] gene={gene} probe={probe} elapsed={elapsed:.1f}s error={agent_error}")
        raise
    elapsed = time.perf_counter() - t0
    log.info(f"[DONE] gene={gene} probe={probe} elapsed={elapsed:.1f}s")
    cost = _calc_cost(result.prompt_tokens, result.completion_tokens, input_cost_per_1m, output_cost_per_1m)

    # Merge structured evidence fields from Pydantic synthesis object.
    # Falls back to conservative defaults if parse() failed (synthesis is None).
    s = result.synthesis
    evidence = {
        "evidence_relation":   s.evidence_relation   if s else "inferred",
        "evidence_tier":       s.evidence_tier       if s else "Tier 4",
        "evidence_confidence": s.evidence_confidence if s else "low",
        "biomarker_potential": s.biomarker_potential if s else "weak",
        "relevance_summary":   s.relevance_summary   if s else "",
    }

    # Deduplicate citations preserving insertion order
    deduped_citations = list(dict.fromkeys(citations))

    payload = {
        **row,
        "model": result.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # Prefer synthesis.interpretation so saved prose always matches parsed output;
        # fall back to result.text only if structured parse failed.
        "interpretation": s.interpretation if s else result.text,
        **evidence,
        "citations": deduped_citations,
        "token_usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "estimated_cost_usd": round(cost, 6),
        },
        "tools_used": [
            {"name": name, "result": res}
            for name, res in (result.used_tools or [])
        ],
    }

    out_path = out_dir / f"{gene}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"[WRITE] gene={gene} path={out_path}")

    return {
        "gene": gene,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost": cost,
        "elapsed": elapsed,
    }


def run(config: dict, biomarker_path: str = None) -> None:
    if biomarker_path is None:
        biomarker_path = config["paths"]["biomarker_shortlist"]

    out_dir = pathlib.Path(config["paths"]["llm_outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_config        = config["llm"]
    model             = llm_config.get("model", DEFAULT_MODEL)
    disease_context   = config.get("project", {}).get("disease", "the studied disease")
    max_workers       = llm_config.get("max_workers", 3)
    input_cost_per_1m  = llm_config.get("input_cost_per_1m", 0.15)
    output_cost_per_1m = llm_config.get("output_cost_per_1m", 0.60)

    # --- Read timeout / rate-limit values from config and apply to tool modules ---
    openai_request_timeout = int(llm_config.get("openai_request_timeout", 300))
    tool_call_sleep        = float(llm_config.get("tool_call_sleep", 0.5))
    gene_timeout           = int(llm_config.get("gene_timeout", 900))
    batch_timeout          = int(llm_config.get("batch_timeout", 7200))
    entrez_timeout         = int(llm_config.get("entrez_timeout", 45))
    entrez_sleep           = float(llm_config.get("entrez_sleep", 0.4))
    wikipedia_timeout      = int(llm_config.get("wikipedia_timeout", 30))

    # Push Entrez settings into Biopython's module-level globals
    from Bio import Entrez as _Entrez
    import app.tools.search_pubmed as _pubmed_mod
    _Entrez.timeout        = entrez_timeout
    _pubmed_mod.ENTREZ_TIMEOUT = entrez_timeout
    _pubmed_mod.ENTREZ_SLEEP   = entrez_sleep

    # --- Runtime introspection: log actual Biopython Entrez state ---
    import socket as _socket
    log.info(
        f"[Entrez config] email={_Entrez.email!r} | api_key={'SET' if getattr(_Entrez, 'api_key', None) else 'NOT SET'}"
        f" | tool={getattr(_Entrez, 'tool', 'NOT PRESENT')!r}"
        f" | max_tries={getattr(_Entrez, 'max_tries', 'NOT PRESENT')}"
        f" | sleep_between_tries={getattr(_Entrez, 'sleep_between_tries', 'NOT PRESENT')}"
    )
    log.info(
        f"[Entrez config] timeout attr on module={getattr(_Entrez, 'timeout', 'NOT PRESENT')}"
        f" | socket.getdefaulttimeout()={_socket.getdefaulttimeout()}"
    )

    # Push Wikipedia timeout (package-version dependent; silent on AttributeError)
    import app.tools.search_wikipedia as _wiki_mod
    _wiki_mod.WIKIPEDIA_TIMEOUT = wikipedia_timeout
    try:
        import wikipedia as _wikipedia_pkg
        _wikipedia_pkg.set_timeout(wikipedia_timeout)
    except AttributeError:
        pass

    genes             = _load_shortlist(biomarker_path)
    top_n_display     = config.get("biomarker", {}).get("top_n_display", 12)
    genes             = genes[:top_n_display]
    log.info(f"LLM batch: processing top {len(genes)} genes (biomarker.top_n_display={top_n_display})")

    # Resume: skip genes that already have output JSON from a prior run
    genes_to_run = [r for r in genes if not (out_dir / f"{r['gene_symbol']}.json").exists()]
    skipped = len(genes) - len(genes_to_run)

    log.info(f"🧠 LLM job: {len(genes_to_run)} genes to run  |  model: {model}  |  workers: {max_workers}")
    if skipped:
        log.info(f"⏭️  Skipping {skipped} already-processed gene(s) — delete JSON to rerun")
    log.info(f"🦠 Disease  : {disease_context}")
    log.info(f"📋 Biomarker Shortlist: {biomarker_path}")
    log.info(f"📁 Output dir: {out_dir}")

    if not genes_to_run:
        log.info("✅ All genes already processed — nothing to do.")
        return

    completed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(f"LLM: {len(genes_to_run)} genes", total=len(genes_to_run))

        for row in genes_to_run:
            gene = row["gene_symbol"]
            try:
                info = _process_gene(
                    row, out_dir, model, disease_context,
                    input_cost_per_1m, output_cost_per_1m,
                    openai_request_timeout, tool_call_sleep,
                )
                completed += 1
                total_prompt_tokens     += info["prompt_tokens"]
                total_completion_tokens += info["completion_tokens"]
                total_cost              += info["cost"]
                progress.advance(task)
                log.info(
                    f"  ✅ {info['gene']}  "
                    f"({info['total_tokens']} tokens  💰 ${info['cost']:.4f}  ⏱ {info['elapsed']:.1f}s)"
                )
            except Exception as e:
                completed += 1
                progress.advance(task)
                log.error(f"  ❌ FAILED for {gene}: {e}")

    total_tokens = total_prompt_tokens + total_completion_tokens
    log.info("=" * 60)
    log.info(f"🏁 LLM job complete — {completed}/{len(genes)} genes processed")
    log.info(f"  🤖 Model            : {model}")
    log.info(f"  📥 Prompt tokens    : {total_prompt_tokens:,}")
    log.info(f"  📤 Completion tokens: {total_completion_tokens:,}")
    log.info(f"  🔢 Total tokens     : {total_tokens:,}")
    log.info(f"  💰 Estimated cost   : ${total_cost:.4f} USD")
    log.info(f"  💲 Pricing used     : ${input_cost_per_1m}/1M input  |  ${output_cost_per_1m}/1M output")
    log.info("=" * 60)
