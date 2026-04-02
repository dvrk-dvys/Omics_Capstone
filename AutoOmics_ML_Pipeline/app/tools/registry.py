import inspect
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Abbreviation reference for all Entrez-backed tools in this project
#
#   NIH     = National Institutes of Health (US government biomedical agency)
#   NCBI    = National Center for Biotechnology Information (division of NIH)
#   Entrez  = NCBI's unified programmatic API for querying its databases
#   PubMed  = NCBI database of biomedical abstracts and citations
#   PMC     = PubMed Central — NCBI's free full-text article archive
#   GEO     = Gene Expression Omnibus — NCBI's genomics dataset repository
#   GDS     = GEO DataSets — the Entrez db name for GEO records
#   UniProt = Universal Protein Resource — curated protein function database
# ---------------------------------------------------------------------------

from app.tools.search_pubmed import PubMedTool
from app.tools.search_wikipedia import WikipediaTool
from app.tools.search_uniprot import UniProtTool
from app.tools.search_opentargets import OpenTargetsTool
from app.tools.search_ncbi_gene import NCBIGeneTool
from app.tools.search_pmc import PMCTool
from app.tools.search_geo import GEOTool

WIKI        = WikipediaTool()
PUBMED      = PubMedTool()
UNIPROT     = UniProtTool()
OPENTARGETS = OpenTargetsTool()
NCBI_GENE   = NCBIGeneTool()
PMC         = PMCTool()
GEO         = GEOTool()


def simple_response_ok(query):
    """
    Signal to the LLM that this is general/non-medical chat
    and it should answer directly without retrieval.

    Returning a small JSON payload is clearer to the llm
    than just the string "OK".
    """
    return {
        "status": "ok",
        "action": "answer_directly",
        "note": "No retrieval needed.",
        "echo_query": query,
    }

FUNCTION_MAP = {
    "pubmed_search":       PUBMED.pubmed_semantic_search,
    "pubmed_fetch_by_id":  PUBMED.pubmed_fetch_by_id,
    "wikipedia_search":    WIKI.wiki_semantic_search,
    "uniprot_search":      UNIPROT.search_uniprot,
    "opentargets_search":  OPENTARGETS.search_opentargets,
    "ncbi_gene_search":    NCBI_GENE.ncbi_gene_search,
    "pmc_fulltext_search": PMC.pmc_fulltext_search,
    "geo_search":          GEO.geo_search,
}

TOOLS_JSON = [
    {
        "type": "function",
        "function": {
            "name": "pubmed_fetch_by_id",
            "description": (
                "Fetch the full abstract for a specific PubMed ID. "
                "Use this to hydrate a PMID cited inside UniProt function text or "
                "Open Targets evidence. One PMID per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "Numeric PubMed ID string, e.g. '10550681'",
                    }
                },
                "required": ["pmid"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Fetch short, relevant Wikipedia passages for definitions/background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed and return top abstract chunks for the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "uniprot_search",
            "description": (
                "Search UniProtKB for a human gene and return its protein function "
                "and known disease associations from the curated Swiss-Prot database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["gene"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "opentargets_search",
            "description": (
                "Query the Open Targets Platform for a gene's function description "
                "and its top disease associations with evidence scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 5,
                    },
                },
                "required": ["gene"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ncbi_gene_search",
            "description": (
                "Search NCBI Gene for a human gene by symbol or name. "
                "Returns the official gene symbol, known aliases/synonyms, and a "
                "functional summary. Use the exact gene symbol as the query (e.g. 'IQGAP1'). "
                "Do NOT include disease context in this query. "
                "Use this in Iter 1 for gene normalization and synonym grounding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pmc_fulltext_search",
            "description": (
                "Search PMC (PubMed Central) full-text articles and return the top semantically "
                "similar passages from article bodies (Methods / Results / Discussion). "
                "Use this when PubMed abstract evidence is weak, missing, or likely incomplete — "
                "PMC full-text recovers gene mentions that only appear in article bodies, "
                "not in PubMed-indexed abstracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geo_search",
            "description": (
                "Search GEO (Gene Expression Omnibus) DataSets for studies matching the query. "
                "Returns dataset-level metadata: title, study type, organism, platform, sample "
                "count, and summary. Use this when dataset or study-level context could strengthen "
                "the interpretation, validate disease/source relevance, or ground expression "
                "findings in publicly available datasets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
