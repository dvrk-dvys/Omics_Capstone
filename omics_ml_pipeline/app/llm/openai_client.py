import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.llm.rag_utils import build_rag_prompt
from openai import OpenAI
from app.tools.registry import FUNCTION_MAP, TOOLS_JSON

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client():
    """Get OpenAI client - lazy instantiation to avoid import-time failures"""
    return OpenAI()


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
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    used_tools: Optional[List[Tuple[str, Any]]] = None


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
    tools_per_iter={0: 2, 1: 1, 2: 1, 3: 1, 4: 0},
) -> LLMResponse:
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
    client = get_openai_client()

    search_results = list(abstracts)  # seed with pre-fetched abstracts
    search_queries = []
    previous_actions = []
    used_tools = []
    iter1_empty = False

    try:
        for i in range(max_iterations):
            print(f"___Iteration {i}___")

            curr_tools_per_iter = (
                tools_per_iter.get(i, 1) if isinstance(tools_per_iter, dict) else tools_per_iter
            )

            if i == 2:
                print()

            try:
                prompt, search_results = build_rag_prompt(
                    gene=gene,
                    search_results=search_results,
                    search_queries=search_queries,
                    previous_actions=previous_actions,
                    max_iter=max_iterations,
                    curr_iter=i,
                )
            except Exception as e:
                print(f"Error building RAG prompt : {e}")
                raise

            messages = []
            messages.append({"role": "system", "content": BIOMARKER_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})

            # Iteration-specific tool policy
            if i == max_iterations - 1:
                # Iter 4 — final synthesis: no tools
                filtered_tools = []
                tool_choice_val = None
            elif i == 0:
                # Iter 0 — structured grounding: both UniProt + Open Targets, always
                filtered_tools = [
                    t for t in tools
                    if t["function"]["name"] in ("uniprot_search", "opentargets_search")
                ]
                tool_choice_val = "required"
            elif i == 1:
                # Iter 1 — literature grounding: contextual PubMed text search
                filtered_tools = [t for t in tools if t["function"]["name"] == "pubmed_search"]
                tool_choice_val = "required"
            elif i == 2:
                # Iter 2 — first enrichment: LLM decides what to follow up on
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
                    # Wikipedia path already covered; skip the LLM call entirely
                    filtered_tools = []
                    tool_choice_val = None
                else:
                    # pubmed_search (repeatable for secondary gene), fetch-by-id, or uniprot again
                    filtered_tools = [
                        t for t in tools
                        if t["function"]["name"] in ("pubmed_search", "pubmed_fetch_by_id", "uniprot_search")
                    ]
                    tool_choice_val = "auto"

            # Skip the API call entirely if no tools are available and this isn't the final iter
            if not filtered_tools and i != max_iterations - 1:
                previous_actions.append(f"ITER:{i}:SKIP")
                continue

            r1 = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=filtered_tools if filtered_tools else None,
                tool_choice=tool_choice_val if filtered_tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            m1 = r1.choices[0].message
            tool_calls = getattr(m1, "tool_calls", None) or []
            tool_calls = tool_calls[:curr_tools_per_iter]

            # Final iteration: always return here — synthesis complete
            if i == max_iterations - 1:
                usage = getattr(r1, "usage", None)
                final_response = LLMResponse(
                    text=m1.content or "",
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    used_tools=used_tools,
                )
                citations = (
                    []
                    if not used_tools or not search_results
                    else [format_citation(d) for d in search_results]
                )
                return final_response, citations

            # Non-final iteration with no tool call: record and advance to next iteration
            if not tool_calls:
                previous_actions.append(f"ITER:{i}:NO_TOOL")
                if i == 1:
                    iter1_empty = True
                continue

            # Track result count before this iteration's tool calls (used for iter1_empty)
            iter_start = len(search_results)

            # Execute tool calls
            tool_messages = []
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                print(f"DEBUG: About to call tool {name} with args: {args}")
                try:
                    result = function_map[name](**args)
                    print(f"DEBUG: Tool {name} returned result type: {type(result)}")
                except Exception as tool_error:
                    print(f"DEBUG: Tool {name} failed with error: {tool_error}")
                    raise  # Re-raise to trigger outer exception handler

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

            # After iter 1: flag whether research tools returned anything useful
            if i == 1:
                iter1_empty = len(search_results) == iter_start

            # feed tool results back
            messages.append(
                {
                    "role": "assistant",
                    "content": m1.content or "",
                    "tool_calls": m1.tool_calls,
                }
            )
            messages.extend(tool_messages)

            print("tool_messages: ", tool_messages)
            print("used_tools: ", used_tools)
            print("search_queries: ", search_queries)
            print("previous_actions: ", previous_actions)

    except Exception as e:
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
    print("🤖 OpenAI Client Test")
    print("=" * 50)

    def in_docker():
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"

    test_genes = [
        # "BPGM",
        "CA1",
        # "GYPA",
        # "RAP1GAP",
    ]

    for gene in test_genes:
        print(gene)
        print("=" * 50)

        # Run the agent loop (4 iterations: uniprot → pubmed+opentargets → conditional → synthesis)
        result, out_citations = agentic_llm(
            gene=gene,
            abstracts=[],   # empty — tools will fetch
            model=DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=500,
        )

        print("— Response —")
        print(result.text)
        print("citations: ", out_citations)

        print("\n— Usage —")
        print(f"Model: {result.model}")
        print(
            f"Prompt tokens: {result.prompt_tokens} | "
            f"Completion tokens: {result.completion_tokens} | "
            f"Total: {result.total_tokens}"
        )

        if result.used_tools:
            print("\n— Tools used —")
            for name, _payload in result.used_tools:
                print(f"- {name}")
        else:
            print("\n— Tools used — none")
        print("=" * 50)
