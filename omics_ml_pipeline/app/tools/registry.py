import inspect
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


from app.tools.search_pubmed import PubMedTool
from app.tools.search_wikipedia import WikipediaTool
from app.tools.search_uniprot import UniProtTool
from app.tools.search_opentargets import OpenTargetsTool

WIKI = WikipediaTool()
PUBMED = PubMedTool()
UNIPROT = UniProtTool()
OPENTARGETS = OpenTargetsTool()


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
    "pubmed_search":      PUBMED.pubmed_semantic_search,
    "pubmed_fetch_by_id": PUBMED.pubmed_fetch_by_id,
    "wikipedia_search":   WIKI.wiki_semantic_search,
    "uniprot_search":     UNIPROT.search_uniprot,
    "opentargets_search": OPENTARGETS.search_opentargets,
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
]
