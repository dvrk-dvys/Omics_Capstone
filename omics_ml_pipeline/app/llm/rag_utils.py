import pathlib
import time
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# GEO dataset context — loaded once at import time
# ---------------------------------------------------------------------------
_input_dir = pathlib.Path(__file__).parent.parent / "data" / "input"
_abstract_files = sorted(_input_dir.glob("*_abstract.txt"))
_abstract_path = _abstract_files[0] if _abstract_files else _input_dir / "abstract.txt"
try:
    geo_dataset_context = _abstract_path.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    geo_dataset_context = "(dataset abstract not found — expected at app/data/input/<dataset>_abstract.txt)"


@contextmanager
def time_block(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = (time.perf_counter() - start) * 1000
        print(f"[TIMER] {label}: {dur:.2f} ms")


# ---------------------------------------------------------------------------
# System prompt — injected once per session / per LLM job run
# ---------------------------------------------------------------------------
OMICS_RESEARCH_SYSTEM_CONTEXT_TEMPLATE = """

ROLE:
You are a biomedical research assistant interpreting candidate biomarkers from a
peripheral blood microarray study. Your job is to assess each gene's biological
relevance to the disease context provided below.

TASK:
- For each CANDIDATE_BIOMARKER you are given, retrieve evidence and produce a
  concise biological interpretation connecting the gene to the study's disease context.
- IMPORTANT: Use the GEO_DATASET_CONTEXT below to understand how the data was
  produced, what the comparison groups are, and how the biomarkers were identified.
- Consult LLM_OUTPUT_HISTORY when cross-referencing previously assessed genes
  (e.g. shared pathways, interacting proteins, consistent signals).
- If evidence is weak or absent, state this and provide your best biological inference
  labeled as [INFERENCE].
Style: plain language first; concise; SI units; define acronyms once.

SYSTEM CONTEXT:
<USER_CONTEXT>
"Research Scientist/ML Engineer": Focus on research findings, methodologies,
statistically valuable results of the baseline ML suite, and scientific evidence
with verifiable citations.
</USER_CONTEXT>

<GEO_DATASET_CONTEXT>
{geo_dataset_context}
</GEO_DATASET_CONTEXT>

- Citations and URL links may only come from TOOL output. Do not fabricate references.

<LLM_OUTPUT_HISTORY>
{llm_output_history}
</LLM_OUTPUT_HISTORY>

""".strip()


# ---------------------------------------------------------------------------
# Per-iteration stage metadata — drives prompt content in build_rag_prompt
# ---------------------------------------------------------------------------
_ITER_STAGES = {
    0: "Iter 0 — Structured Grounding",
    1: "Iter 1 — Literature Grounding",
    2: "Iter 2 — First Enrichment",
    3: "Iter 3 — Second Enrichment",
    4: "Iter 4 — Final Synthesis",
}

_ITER_GUIDANCE = {
    0: (
        "Both uniprot_search AND opentargets_search are required this iteration.\n"
        "- uniprot_search: retrieve canonical protein function, subcellular location, relevant pathways.\n"
        "- opentargets_search: retrieve disease-gene associations and confidence scores.\n"
        "Call both tools. Their combined output frames all subsequent retrieval and synthesis."
    ),
    1: (
        "pubmed_search is required this iteration.\n"
        "- Formulate ONE narrow PubMed query for the candidate gene.\n"
        "- The first PubMed query must use the exact gene symbol and the exact disease name.\n"
        "- Prefer a literal disease query such as: '<GENE> {{disease_context}}'.\n"
        "- Do NOT broaden the first query with generic phrases like 'role in', 'bone health', "
        "'signaling', 'mechanism', or broad OR clauses.\n"
        "- Do NOT use OR in the first PubMed query unless a synonym is explicitly present in retrieved tool output.\n"
        "- The goal of Iter 1 is to test for direct literature grounding first, not to maximize recall.\n"
        "- If the exact disease query returns weak or no results, later iterations may broaden carefully."
    ),
    2: (
        "Optional follow-up — choose ONE branch based on Iter 0–1 results, or skip:\n"
        "  Branch A: if Iter 1 returned no results or very weak results\n"
        "            → call wikipedia_search as a last resort.\n"
        "  Branch B: if a biologically relevant secondary/interacting gene was identified\n"
        "            → call uniprot_search(secondary_gene).\n"
        "  Branch C: if a specific high-value PMID was cited in retrieved results\n"
        "            → call pubmed_fetch_by_id(pmid).\n"
        "  Branch D: if retrieved context is already sufficient for a strong interpretation\n"
        "            → do not call a tool."
    ),
    3: (
        "Optional final follow-up — choose ONE or skip:\n"
        "- If Iter 2 retrieved a secondary gene via uniprot_search\n"
        "  → follow up with pubmed_search (secondary gene + {{disease_context}} context)\n"
        "    or pubmed_fetch_by_id if a specific PMID is worth hydrating.\n"
        "- If Iter 2 already hydrated a specific PMID → skip unless new leads emerged.\n"
        "- If Iter 2 used wikipedia fallback → skip unless something actionable was surfaced.\n"
        "- If context is already sufficient → do not call a tool."
    ),
    4: (
        "No tools this iteration. Produce the final interpretation now.\n"
        "Cover all four points in OUTPUT FORMAT below.\n"
        "Distinguish direct evidence (from retrieved sources) from [INFERENCE].\n"
        "End with a one-sentence assessment of blood-based early-detection potential."
    ),
}

_ITER_REASONING = {
    0: (
        "You must call both uniprot_search AND opentargets_search. "
        "Do not attempt to write an interpretation yet."
    ),
    1: (
        "You must call pubmed_search. "
        "Use a conservative retrieval strategy: start with the narrowest exact disease query first. "
        "Prefer exact gene + exact disease wording over broader mechanistic phrasing. "
        "Do not write an interpretation yet."
    ),
    2: (
        "Review PREVIOUS_ACTIONS. "
        "Decide which branch applies (wikipedia fallback / secondary gene lookup / "
        "PMID hydration / skip). Use at most ONE tool."
    ),
    3: (
        "Review PREVIOUS_ACTIONS. "
        "Decide if one more follow-up adds meaningful new evidence. "
        "Skip if context is already sufficient for synthesis."
    ),
    4: "No tools available. Write the final synthesis now.",
}


# ---------------------------------------------------------------------------
# Per-iteration user-turn prompt template
# ---------------------------------------------------------------------------
OMICS_AGENTIC_PROMPT_TEMPLATE = """
TASK:
Assess the biological and clinical relevance of the CANDIDATE_BIOMARKER below to
{disease_context}. Retrieve evidence across the iteration loop, then synthesize
a final interpretation.

<CANDIDATE_BIOMARKER>
{gene}
</CANDIDATE_BIOMARKER>

<RETRIEVED_CONTEXT>
{context}
</RETRIEVED_CONTEXT>

You can perform the following actions:

- ANSWER_CONTEXT : synthesize the biomarker interpretation using RETRIEVED_CONTEXT.
- ANSWER         : provide interpretation from your own biomedical knowledge when context is insufficient.
- TOOL           : select ONE available tool to retrieve additional evidence.

TOOLS (available this iteration):

<TOOLS>
{tools}
</TOOLS>

CURRENT ITERATION: {curr_iter} of {max_iter} — {iteration_stage}

<ITERATION_GUIDANCE>
{iteration_guidance}
</ITERATION_GUIDANCE>

REASONING (internal — do not expose):
{reasoning_guidance}

OUTPUT FORMAT:
- If calling a tool: output only the tool call. Do not produce a narrative answer yet.
- If this is the FINAL SYNTHESIS iteration (Iter 4), produce a 4–8 sentence interpretation:
    (a) What the gene/protein does biologically.
    (b) Its connection to the known mechanisms of {disease_context}
        (as described in the GEO_DATASET_CONTEXT in the system prompt).
    (c) Strength of evidence — label clearly: direct / indirect / [INFERENCE].
    (d) Clinical relevance as an early-detection biomarker for {disease_context}.

STOP CONDITIONS:
- Stop after the final synthesis iteration. Do not continue beyond iteration {max_iter}.
- If no tool is needed this iteration, do not call one.
- If prior tool results are already sufficient, proceed to synthesis without calling more tools.
- If evidence is weak, state this explicitly and provide your best biological [INFERENCE].

RULES:
- Maximum {max_iter} iterations total. This is iteration {curr_iter}.
- Do NOT fabricate citations or PMIDs
- Do NOT repeat a tool that already appears in PREVIOUS_ACTIONS,
  except uniprot_search which may repeat for a secondary/interacting gene.
- Select at most ONE tool per iteration, unless the ITERATION_GUIDANCE explicitly requires more.
- Retrieved PubMed abstracts are primary evidence — do not cite papers not supplied by tools and do not recall training data as evidence.
- Label all biological reasoning not backed by a retrieved source as [INFERENCE].
- If evidence is weak or absent, explicitly state that
- In Iter 1, the first PubMed query must be narrow and disease-literal.
- In Iter 1, avoid broad phrases such as 'role in', 'bone health', 'mechanism', or generic pathway language.
- In Iter 1, avoid OR clauses unless a synonym is explicitly grounded in prior retrieved context.
- Only broaden the query in later iterations if the narrow disease query produced weak or no evidence.


<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>

""".strip()


# ---------------------------------------------------------------------------
# Helper: format LLM output history for system prompt injection
# ---------------------------------------------------------------------------
def _format_llm_output_history(llm_output_history: list) -> str:
    """
    Format a list of per-gene LLM output payloads (from llm_job.py) into a
    readable string for injection into the system prompt.

    Each entry is expected to have at minimum:
      gene_symbol, interpretation, citations (list[str])
    Optional fields used if present: combined_score
    """
    if not llm_output_history:
        return "(no prior gene interpretations available)"

    lines = []
    for entry in llm_output_history:
        gene = entry.get("gene_symbol", "UNKNOWN")
        interp = (entry.get("interpretation") or "").strip()
        citations = entry.get("citations") or []
        score = entry.get("combined_score")

        header = f"--- Gene: {gene}"
        if score is not None:
            header += f"  (combined_score: {score})"
        header += " ---"
        lines.append(header)

        if interp:
            lines.append(interp)

        if citations:
            lines.append("Citations:")
            for c in citations:
                lines.append(f"  - {c}")

        lines.append("")  # blank line between entries

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_rag_context(llm_output_history=None):
    """
    Build the system context string for the omics agentic LLM.

    Parameters
    ----------
    llm_output_history : list[dict] | None
        List of per-gene LLM output payloads produced by llm_job.py.
        Each dict contains: gene_symbol, interpretation, citations,
        combined_score, probe_id, rf_importance, etc.
        Pass the accumulated list after each gene is processed so the LLM
        can cross-reference earlier interpretations while answering queries.
    """
    formatted_history = _format_llm_output_history(llm_output_history)

    return OMICS_RESEARCH_SYSTEM_CONTEXT_TEMPLATE.format(
        geo_dataset_context=geo_dataset_context,
        llm_output_history=formatted_history,
    )


def build_rag_prompt(
    gene,
    search_results,
    search_queries,
    previous_actions,
    max_iter,
    curr_iter,
    tools=None,
    disease_context="the studied disease",
):
    """
    Build the user-turn prompt for a single agentic iteration.

    Parameters
    ----------
    gene            : str   — gene symbol being assessed (e.g. "BPGM")
    search_results  : list  — accumulated retrieved docs from all prior tool calls
    search_queries  : list  — query strings sent so far
    previous_actions: list  — "ITER:N:TOOL:name(args)" log
    max_iter        : int   — total iterations in the loop
    curr_iter       : int   — 0-based current iteration index
    tools           : list  — tool dicts available this iteration (for prompt display);
                              if None, the TOOLS section says "(none — final synthesis)"
    disease_context : str   — short disease name/acronym injected into the prompt
                              (e.g. "Steroid-Induced Osteonecrosis of the Femoral Head (SONFH)").
                              Comes from config["project"]["disease"] so swapping datasets
                              only requires updating pipeline.yaml.
    """
    # Format retrieved context
    context_text = "\n\n".join(
        f"[{d.get('source_type', '?')}] {d.get('title', '?')}\n{d.get('text', '')}"
        for d in search_results
    ) or "(no context retrieved yet)"

    # Build readable tool list for the prompt
    if tools:
        tool_lines = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in tools
            if t.get("type") == "function" and "function" in t
        )
    else:
        tool_lines = "(none — produce final synthesis)"

    # Iteration-specific content
    # Guidance strings for iter 1 and iter 3 contain {disease_context} placeholders
    # (written as {{disease_context}} in the dict literals to survive the outer .format call)
    stage = _ITER_STAGES.get(curr_iter, f"Iter {curr_iter}")
    guidance = _ITER_GUIDANCE.get(curr_iter, "Use available tools or synthesize.").format(
        disease_context=disease_context
    )
    reasoning = _ITER_REASONING.get(curr_iter, "Decide whether to call a tool or synthesize.")

    prompt = OMICS_AGENTIC_PROMPT_TEMPLATE.format(
        gene=gene,
        context=context_text,
        tools=tool_lines,
        curr_iter=curr_iter + 1,      # 1-based for readability in prompt
        max_iter=max_iter,
        iteration_stage=stage,
        iteration_guidance=guidance,
        reasoning_guidance=reasoning,
        disease_context=disease_context,
        search_queries="\n".join(search_queries) if search_queries else "(none yet)",
        previous_actions="\n".join(previous_actions) if previous_actions else "(none yet)",
    )

    return prompt, search_results


# ---------------------------------------------------------------------------
# Quick test harness — eyeball both prompts across all 5 iterations
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    GENE    = "BPGM"
    DISEASE = "Steroid-Induced Osteonecrosis of the Femoral Head (SONFH)"

    # --- Sample retrieved docs (what tools would return) ---
    sample_results = [
        {
            "source_type": "uniprot",
            "title": "BPGM — Bisphosphoglycerate Mutase (UniProt P07363)",
            "text": (
                "Catalyzes the interconversion of 2,3-bisphosphoglycerate (2,3-BPG). "
                "Expressed almost exclusively in erythrocytes. 2,3-BPG is a key allosteric "
                "regulator of hemoglobin oxygen affinity — high 2,3-BPG shifts the O2 "
                "dissociation curve rightward, facilitating oxygen release to tissues."
            ),
        },
        {
            "source_type": "pubmed",
            "title": "Erythrocyte 2,3-BPG depletion and impaired oxygen delivery in bone ischemia (2021)",
            "text": (
                "Reduced BPGM expression in peripheral blood correlates with decreased 2,3-BPG "
                "levels and impaired oxygen delivery to hypoxic bone tissue. Patients with "
                "avascular necrosis showed a 3.4-fold reduction in BPGM mRNA vs controls."
            ),
            "year": "2021",
            "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/",
        },
    ]

    # --- Sample LLM output history (a prior gene already processed) ---
    sample_history = [
        {
            "gene_symbol": "GYPA",
            "combined_score": 0.791,
            "interpretation": (
                "Glycophorin A (GYPA) is the major RBC surface sialoglycoprotein and a canonical "
                "marker of erythroid lineage. Its consistent downregulation across all CV folds in "
                "SONFH blood is consistent with a systemic erythroid suppression pattern, possibly "
                "reflecting impaired erythropoiesis or accelerated RBC clearance under chronic "
                "corticosteroid exposure."
            ),
            "citations": [
                "Erythroid lineage markers in steroid-induced bone disease (2022). "
                "https://pubmed.ncbi.nlm.nih.gov/88888888/"
            ],
        }
    ]

    # Simulated tool availability per iteration (mirrors openai_client.py policy)
    _ITER_MOCK_TOOLS = {
        0: ["uniprot_search", "opentargets_search"],
        1: ["pubmed_search"],
        2: ["uniprot_search", "pubmed_fetch_by_id"],
        3: ["pubmed_search", "pubmed_fetch_by_id", "uniprot_search"],
        4: [],
    }

    # --- System prompt ---
    print("\n================= SYSTEM PROMPT =================\n")
    with time_block("build_rag_context"):
        sys_prompt = build_rag_context(llm_output_history=sample_history)
    print(sys_prompt)

    # --- User prompt for each iteration ---
    for iter_idx in range(5):
        mock_tools = [
            {"type": "function", "function": {"name": n, "description": f"(mock) {n}"}}
            for n in _ITER_MOCK_TOOLS.get(iter_idx, [])
        ]
        prev_actions = [
            f"ITER:{i}:TOOL:mock_tool({{}})" for i in range(iter_idx)
        ]

        with time_block(f"build_rag_prompt  iter={iter_idx}"):
            user_prompt, _ = build_rag_prompt(
                gene=GENE,
                search_results=sample_results,
                search_queries=["BPGM osteonecrosis", "bisphosphoglycerate mutase bone ischemia"],
                previous_actions=prev_actions,
                max_iter=5,
                curr_iter=iter_idx,
                tools=mock_tools,
                disease_context=DISEASE,
            )

        print(f"\n{'=' * 60}")
        print(f"ITER {iter_idx} — {_ITER_STAGES.get(iter_idx)}")
        print(f"{'=' * 60}\n")
        print(user_prompt)
