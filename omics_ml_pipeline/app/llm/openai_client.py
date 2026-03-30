import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.llm.rag_utils import build_rag_context, build_rag_prompt, BiomarkerSynthesis
from openai import OpenAI
from app.tools.registry import FUNCTION_MAP, TOOLS_JSON

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client(timeout: int = 300):
    """Get OpenAI client with a per-request timeout (default 300 s)."""
    return OpenAI(timeout=timeout)


DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
available_models = ["gpt-4o-mini", "gpt-4o", "o3", "o4"]


BIOMARKER_SYSTEM_PROMPT = """
You are a biomedical researcher interpreting gene expression results from a peripheral blood
microarray study of Steroid-Induced Osteonecrosis of the Femoral Head (SONFH) vs
steroid-treated controls (GSE123568, n=40, Affymetrix PrimeView GPL15207).

Study context: 30 SONFH patients vs 10 non-SONFH steroid controls — both groups received
corticosteroids; the comparison isolates disease susceptibility from steroid exposure alone.
The clinical goal is early blood-based biomarkers for pre-collapse detection of SONFH.

Known disease mechanisms from recent literature:
- Coagulopathy: 84.7% of ONFH patients show abnormal coagulation profiles (hyperfibrinogen,
  elevated D-dimer, abnormal protein S/C, reduced anti-thrombin III); thromboembolism in the
  femoral head microvasculature is a primary pathway to ischemia and bone death.
- Immune dysregulation: steroid-induced ONFH shows a neutrophil-predominant innate immune
  signature, impaired T cell maturation, and exhausted CD8+ T cells — distinct from
  alcohol-induced ONFH and hip osteoarthritis.
- Vascular disruption: increased pericytes and endothelial cells with altered stromal
  composition have been observed in steroid-induced ONFH femoral head tissue.
- Shared etiology: both steroid use and alcohol abuse trigger osteonecrosis through overlapping
  vascular disruption pathways; literature on alcohol-induced ONFH is relevant and transferable.

Your task: for each candidate biomarker gene, explain its biological relevance to SONFH using
retrieved PubMed evidence, and use your own biological reasoning to connect findings to the
mechanisms above.

Rules:
- Retrieved PubMed abstracts are your primary evidence — do NOT cite papers not supplied.
- You MAY use your own biological knowledge to expand and explain connections between the
  retrieved literature and our specific omics findings — label such reasoning as inference,
  not citation.
- If evidence is weak or tangentially related, say so explicitly.
- Search broadly across these dimensions:
    • Direct SONFH / osteonecrosis / avascular necrosis connections
    • Corticosteroid effects on bone, vasculature, lipid metabolism, and erythrocytes
    • Alcohol-related osteonecrosis (overlapping vascular disruption mechanism)
    • Coagulopathy, thromboembolism, fibrinogen, D-dimer, protein S/C, factor V
    • Immune dysregulation: neutrophil biology, T cell dysfunction, innate/adaptive imbalance
    • Hypoxia, ischemia, and impaired oxygen delivery to bone
    • Erythrocyte biology and peripheral blood transcriptome changes
    • Vascular endothelial and pericyte dysfunction
""".strip()


@dataclass
class LLMResponse:
    text: str                                        # prose interpretation (backward-compatible)
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    used_tools: Optional[List[Tuple[str, Any]]] = None
    synthesis: Optional[BiomarkerSynthesis] = None   # structured Iter 4 output; None for earlier iters


def llm(prompt, sys_prompt=None, model=DEFAULT_MODEL) -> LLMResponse:
    """
    Call OpenAI LLM with the prompt

    "system" – instructions/behavior setup.
    "user" – the user's prompt.
    "assistant" – previous assistant replies (for context).
    "tool" – the output returned by a tool/function you called (if you use tools).
    """

    try:
        messages = []

        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=500
        )

        msg = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tok = getattr(usage, "total_tokens", prompt_tok + completion_tok)

        return LLMResponse(
            text=msg,
            model=model,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=total_tok,
        )

    except Exception as e:
        return LLMResponse(
            text=f"Error calling LLM: {e}",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


def filter_used_tools(tools, used, repeatable=None):
    """Return tools list with already used tool names removed.
    Tools named in `repeatable` are always kept regardless of prior use."""
    repeatable = set(repeatable or [])
    used_names = {name for name, _ in used}
    return [
        t for t in tools
        if t["function"]["name"] in repeatable
        or t["function"]["name"] not in used_names
    ]


def format_citation(doc):
    stype = doc.get("source_type")
    title = doc.get("title", "Unknown")

    url = doc.get("url", "")
    if stype == "pubmed":
        year = doc.get("year", "")
        return f"{title} ({year}). {url}"
    elif stype in ("wikipedia", "uniprot", "opentargets"):
        return f"{title}. {url}"
    else:
        return title


def agentic_llm(
    gene,
    abstracts,
    tools=TOOLS_JSON,
    function_map=FUNCTION_MAP,
    model=DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=500,
    max_iterations=5,
    tools_per_iter=None,
    disease_context="the studied disease",
    llm_output_history=None,
    openai_request_timeout: int = 300,
    tool_call_sleep: float = 0.5,
) -> Tuple[LLMResponse, List[str]]:
    """
    Agentic LLM loop for a single biomarker gene. Runs up to 4 iterations.

    Iteration policy:
      Iter 0 — Structured grounding : force uniprot_search + opentargets_search (required, cap=2)
      Iter 1 — Literature grounding : force pubmed_search with contextual query (required, cap=1)
      Iter 2 — First enrichment     : auto, cap=1
                Branch A: PubMed returned nothing  → wikipedia_search fallback
                Branch B/C: secondary gene found   → uniprot_search(secondary_gene)
                Branch B/C: specific PMID to hydrate → pubmed_fetch_by_id(pmid)
                Branch D: enough info already       → no tool call, skip to Iter 3
      Iter 3 — Second enrichment    : auto, cap=1
                if iter1_empty: skip (nothing to follow up on)
                otherwise: pubmed_search / pubmed_fetch_by_id / uniprot_search — LLM decides
      Iter 4 — Final synthesis      : no tools; collate all evidence, connect to GSE123568

    State accumulated across iterations:
      search_results   – all retrieved passages from every tool call
      search_queries   – query strings sent to each tool
      previous_actions – "ITER:N:TOOL:name(args)" log fed back into the prompt
      used_tools       – (name, result) list; uniprot_search is repeatable, others are one-shot
    """
    if tools_per_iter is None:
        tools_per_iter = {0: 2, 1: 1, 2: 1, 3: 1, 4: 0}

    client = get_openai_client(timeout=openai_request_timeout)

    search_results = list(abstracts)  # seed with pre-fetched abstracts
    search_queries = []
    previous_actions = []
    used_tools = []
    iter1_empty = False

    try:
        for i in range(max_iterations):
            curr_tools_per_iter = (
                tools_per_iter.get(i, 1) if isinstance(tools_per_iter, dict) else tools_per_iter
            )

            # ------------------------------------------------------------------
            # Iteration-specific tool policy  (must be resolved before prompt)
            # ------------------------------------------------------------------
            if i == max_iterations - 1:
                # Iter 4 — final synthesis: no tools
                filtered_tools = []
                tool_choice_val = None
            elif i == 0:
                # Iter 0 — structured grounding: UniProt + Open Targets, required
                filtered_tools = [
                    t for t in tools
                    if t["function"]["name"] in ("uniprot_search", "opentargets_search")
                ]
                tool_choice_val = "required"
            elif i == 1:
                # Iter 1 — literature grounding: contextual PubMed search, required
                filtered_tools = [t for t in tools if t["function"]["name"] == "pubmed_search"]
                tool_choice_val = "required"
            elif i == 2:
                # Iter 2 — first enrichment: LLM decides
                if iter1_empty:
                    # PubMed came back empty — Wikipedia as last resort
                    filtered_tools = [t for t in tools if t["function"]["name"] == "wikipedia_search"]
                else:
                    # Secondary gene lookup OR hydrate a specific cited PMID
                    filtered_tools = [
                        t for t in tools
                        if t["function"]["name"] in ("uniprot_search", "pubmed_fetch_by_id")
                    ]
                tool_choice_val = "auto"
            else:
                # Iter 3 — second enrichment: one final follow-up, LLM decides
                if iter1_empty:
                    # Wikipedia path already covered; skip entirely
                    filtered_tools = []
                    tool_choice_val = None
                else:
                    filtered_tools = [
                        t for t in tools
                        if t["function"]["name"] in ("pubmed_search", "pubmed_fetch_by_id", "uniprot_search")
                    ]
                    tool_choice_val = "auto"

            # ------------------------------------------------------------------
            # Iteration start — log resolved tool list and start wall timer
            # ------------------------------------------------------------------
            _iter_t0 = time.perf_counter()
            _allowed = [t["function"]["name"] for t in filtered_tools] if filtered_tools else []
            print(f"[ITER {i}] starting | gene={gene} | tools_allowed={_allowed}", flush=True)

            # Skip the API call entirely if no tools and not the final synthesis iter
            if not filtered_tools and i != max_iterations - 1:
                previous_actions.append(f"ITER:{i}:SKIP")
                print(f"[ITER {i}] SKIP (no tools, not final) elapsed={time.perf_counter() - _iter_t0:.2f}s", flush=True)
                continue

            # ------------------------------------------------------------------
            # Build prompt — filtered_tools now resolved so display is accurate
            # ------------------------------------------------------------------
            sys_prompt = build_rag_context(llm_output_history)

            try:
                prompt, search_results = build_rag_prompt(
                    gene=gene,
                    search_results=search_results,
                    search_queries=search_queries,
                    previous_actions=previous_actions,
                    max_iter=max_iterations,
                    curr_iter=i,
                    tools=filtered_tools,
                    disease_context=disease_context,
                )
            except Exception as e:
                print(f"Error building RAG prompt : {e}", flush=True)
                raise

            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})

            print(f"[ITER {i}] calling LLM...", flush=True)

            # ------------------------------------------------------------------
            # Iter 4 — final synthesis: structured output via parse(), returns here.
            # Iters 0–3 fall through to the standard create() + tool-call path below.
            # ------------------------------------------------------------------
            if i == max_iterations - 1:
                r1        = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=BiomarkerSynthesis,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                m1        = r1.choices[0].message
                synthesis = getattr(m1, "parsed", None)   # BiomarkerSynthesis | None
                if synthesis is None:
                    print(
                        f"[ITER {i}] WARNING: structured parse failed, falling back to raw text",
                        flush=True,
                    )
                print(
                    f"[ITER {i}] FINAL SYNTHESIS (structured) parsed={synthesis is not None}",
                    flush=True,
                )
                usage = getattr(r1, "usage", None)
                final_response = LLMResponse(
                    text=synthesis.interpretation if synthesis else (m1.content or ""),
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    used_tools=used_tools,
                    synthesis=synthesis,
                )
                citations = (
                    []
                    if not used_tools or not search_results
                    else [format_citation(d) for d in search_results]
                )
                print(f"[ITER {i}] DONE elapsed={time.perf_counter() - _iter_t0:.2f}s", flush=True)
                return final_response, citations

            # ------------------------------------------------------------------
            # Iters 0–3 — standard create() with tool-calling (unchanged)
            # ------------------------------------------------------------------
            r1 = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=filtered_tools if filtered_tools else None,
                tool_choice=tool_choice_val if filtered_tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(f"[ITER {i}] LLM response received", flush=True)
            m1 = r1.choices[0].message
            tool_calls = getattr(m1, "tool_calls", None) or []
            tool_calls = tool_calls[:curr_tools_per_iter]
            if tool_calls:
                print(f"[ITER {i}] tool_calls={len(tool_calls)}", flush=True)

            # Non-final iteration with no tool call: record and advance to next iteration
            if not tool_calls:
                print(f"[ITER {i}] NO TOOL CALL", flush=True)
                previous_actions.append(f"ITER:{i}:NO_TOOL")
                if i == 1:
                    iter1_empty = True
                    print(f"[ITER {i}] iter1_empty=True (no tool call)", flush=True)
                print(f"[ITER {i}] DONE elapsed={time.perf_counter() - _iter_t0:.2f}s", flush=True)
                continue

            # Track result count before this iteration's tool calls (used for iter1_empty)
            iter_start = len(search_results)

            # Execute tool calls
            tool_messages = []
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                print(f"[ITER {i}] TOOL START: {name} args={args}", flush=True)
                _t_tool = time.perf_counter()
                try:
                    result = function_map[name](**args)
                except Exception as tool_error:
                    _tool_elapsed = time.perf_counter() - _t_tool
                    print(
                        f"[ITER {i}] TOOL FAIL: gene={gene} tool={name} "
                        f"args={args} elapsed={_tool_elapsed:.2f}s error={tool_error}",
                        flush=True,
                    )
                    previous_actions.append(
                        f"ITER:{i}:TOOL_FAIL:{name}({args}) error={tool_error}"
                    )
                    time.sleep(tool_call_sleep)
                    continue

                _tool_elapsed = time.perf_counter() - _t_tool
                print(f"[ITER {i}] TOOL DONE: {name} elapsed={_tool_elapsed:.2f}s", flush=True)

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                used_tools.append((name, result))
                search_queries.append(str(args))
                previous_actions.append(f"ITER:{i}:TOOL:{name}({args})")

                if isinstance(result, list):
                    search_results.extend(result)
                elif isinstance(result, dict) and "hits" in result:
                    search_results.extend(result["hits"])

                time.sleep(tool_call_sleep)

            print(f"[ITER {i}] search_results_count={len(search_results)}", flush=True)

            # After iter 1: flag whether research tools returned anything useful
            if i == 1:
                iter1_empty = len(search_results) == iter_start
                print(f"[ITER {i}] iter1_empty={iter1_empty}", flush=True)

            # feed tool results back
            messages.append(
                {
                    "role": "assistant",
                    "content": m1.content or "",
                    "tool_calls": m1.tool_calls,
                }
            )
            messages.extend(tool_messages)
            print(f"[ITER {i}] DONE elapsed={time.perf_counter() - _iter_t0:.2f}s", flush=True)

    except Exception as e:
        print(f"[AGENT FAIL] gene={gene} error={e}", flush=True)
        return (
            LLMResponse(
                text=f"Error calling LLM: {e}",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                used_tools=None,
            ),
            [],
        )


if __name__ == "__main__":
    import pathlib
    from datetime import datetime

    # Smoke test: run the full agentic loop for one gene and verify the new
    # structured output contract (BiomarkerSynthesis) round-trips correctly.
    BIOMARKER = {
        "probe_id": "11758559_s_at",
        "gene_symbol": "NUDT4",
        "rf_importance": 0.0512,
        "selection_freq": 1.0,
        "abs_fold_change": 2.1,
        "combined_score": 0.8363,
    }

    gene = BIOMARKER["gene_symbol"]
    disease = "Steroid-Induced Osteonecrosis of the Femoral Head (SONFH)"
    print("=" * 60)
    print(f"SMOKE TEST — agentic_llm + structured output")
    print(f"Gene  : {gene}  |  probe: {BIOMARKER['probe_id']}  |  score: {BIOMARKER['combined_score']}")
    print(f"Model : {DEFAULT_MODEL}")
    print("=" * 60)

    t_total = time.perf_counter()
    result, out_citations = agentic_llm(
        gene=gene,
        abstracts=[],
        model=DEFAULT_MODEL,
        temperature=0.1,
        max_tokens=500,
        disease_context=disease,
    )
    elapsed = time.perf_counter() - t_total
    print(f"\nagentic_llm runtime : {elapsed:.2f}s")

    # --- Structured output check ---
    s = result.synthesis
    if s is not None:
        print("\n[PASS] Structured synthesis parsed successfully")
        print(f"  evidence_relation   : {s.evidence_relation}")
        print(f"  evidence_tier       : {s.evidence_tier}")
        print(f"  evidence_confidence : {s.evidence_confidence}")
        print(f"  biomarker_potential : {s.biomarker_potential}")
        print(f"  relevance_summary   : {s.relevance_summary}")
    else:
        print("\n[WARN] Structured parse returned None — fallback to raw text")

    # --- Build payload matching llm_job.py contract ---
    evidence = {
        "evidence_relation":   s.evidence_relation   if s else "inferred",
        "evidence_tier":       s.evidence_tier       if s else "Tier 4",
        "evidence_confidence": s.evidence_confidence if s else "low",
        "biomarker_potential": s.biomarker_potential if s else "weak",
        "relevance_summary":   s.relevance_summary   if s else "",
    }
    deduped_citations = list(dict.fromkeys(out_citations))

    output = {
        **BIOMARKER,
        "model": result.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "interpretation": s.interpretation if s else result.text,
        **evidence,
        "citations": deduped_citations,
        "token_usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        "tools_used": [
            {"name": name, "result": res}
            for name, res in (result.used_tools or [])
        ],
    }

    # Save to app/data/output/llm_outputs/{gene}.json
    out_dir = pathlib.Path(__file__).parent.parent / "data" / "output" / "llm_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{gene}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # --- Print summary ---
    print("\n— Interpretation —")
    print(s.interpretation if s else result.text)

    print(f"\n— Citations ({len(deduped_citations)}) —")
    for c in deduped_citations:
        print(f"  {c}")

    print(f"\n— Token usage —")
    print(
        f"  Prompt: {result.prompt_tokens} | "
        f"Completion: {result.completion_tokens} | "
        f"Total: {result.total_tokens}"
    )

    if result.used_tools:
        print(f"\n— Tools used ({len(result.used_tools)}) —")
        for name, _ in result.used_tools:
            print(f"  - {name}")
    else:
        print("\n— Tools used — none")

    # Verify all expected keys are present in the saved JSON
    expected_keys = {
        "interpretation", "evidence_relation", "evidence_tier",
        "evidence_confidence", "biomarker_potential", "relevance_summary",
        "citations", "token_usage", "tools_used",
    }
    missing = expected_keys - set(output.keys())
    if missing:
        print(f"\n[FAIL] Output JSON missing keys: {missing}")
    else:
        print(f"\n[PASS] All expected output keys present")

    print(f"\n— Saved → {out_path}")
    print(f"Total runtime       : {elapsed:.2f}s")
