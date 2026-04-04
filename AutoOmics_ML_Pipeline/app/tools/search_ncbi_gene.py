import time

from Bio import Entrez

from app.tools.entrez_utils import entrez_esearch, entrez_esummary


class NCBIGeneTool:
    """
    NCBI Gene database lookup via entrez_esearch + entrez_esummary.

    Abbreviation context:
      NIH   = National Institutes of Health (US government biomedical agency)
      NCBI  = National Center for Biotechnology Information (division of NIH)
      Entrez = NCBI's unified API for querying its databases (Gene, PubMed, GEO, PMC, etc.)

    NCBI Gene is a curated database of gene-level metadata for organisms across
    all kingdoms.  For each gene entry it stores: the official gene symbol,
    known aliases and synonyms, a plain-language functional summary, chromosomal
    location, organism taxonomy, and cross-references to other NCBI resources.

    Why this tool is used in Iter 1:
      PubMed literature is indexed using a mix of official symbols, historical
      aliases, and protein names.  A query for "IQGAP1" may miss papers that
      refer to the same gene as "HUMORFA01" or "IQGAP".  By fetching the NCBI
      Gene record first, the pipeline establishes:
        1. The authoritative official symbol.
        2. All known aliases and synonyms.
        3. A functional summary of what the gene/protein does biologically.
      This grounding allows later retrieval (PubMed, PMC, GEO) to use the
      correct vocabulary and to interpret alias-based mentions correctly.

    All queries in this tool are restricted to Homo sapiens by default.

    Result source_type: "ncbi_gene"
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gene_url(gene_id: str) -> str:
        return f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"

    def _search_gene_ids(self, query: str, retmax: int = 5) -> list:
        """NCBI Gene esearch → list of gene ID strings, Homo sapiens only."""
        term = f"{query} AND Homo sapiens[Organism]"
        try:
            handle = entrez_esearch(
                db="gene",
                term=term,
                retmode="xml",
                retmax=retmax,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return list(record.get("IdList", []))
        except Exception as e:
            print(f"[NCBIGene] esearch error: {e}")
            return []

    def _fetch_gene_summaries(self, gene_ids: list) -> list:
        """NCBI Gene esummary for a list of IDs → list of DocSum dicts."""
        if not gene_ids:
            return []
        ids_str = ",".join(str(i) for i in gene_ids)
        try:
            handle  = entrez_esummary(db="gene", id=ids_str)
            records = Entrez.read(handle)
            handle.close()
            # Biopython may return a flat list or a DocumentSummarySet wrapper
            if isinstance(records, list):
                return list(records)
            if hasattr(records, "get"):
                inner = records.get("DocumentSummarySet", {})
                if hasattr(inner, "get"):
                    return list(inner.get("DocumentSummary", []))
            return []
        except Exception as e:
            print(f"[NCBIGene] esummary error: {e}")
            return []

    @staticmethod
    def _format_hit(rec) -> dict:
        """Format one NCBI Gene DocSum record into a compact RAG-friendly dict."""
        gene_id = str(rec.get("Id", rec.get("uid", ""))).strip()
        symbol  = str(rec.get("Name",         "")).strip()
        desc    = str(rec.get("Description",  "")).strip()
        aliases = str(rec.get("OtherAliases", "")).strip()
        summary = str(rec.get("Summary",      "")).strip()

        organism_node = rec.get("Organism", {})
        organism = (
            str(organism_node.get("ScientificName", "")).strip()
            if hasattr(organism_node, "get")
            else str(organism_node).strip()
        )

        parts = []
        if symbol:
            parts.append(f"Official symbol: {symbol}")
        if desc:
            parts.append(f"Gene name: {desc}")
        if aliases:
            parts.append(f"Aliases / synonyms: {aliases}")
        if organism:
            parts.append(f"Organism: {organism}")
        if summary:
            trunc = summary[:800] + ("..." if len(summary) > 800 else "")
            parts.append(f"Gene summary: {trunc}")

        title = f"{symbol} — {desc}" if (symbol and desc) else (symbol or desc or gene_id)
        text  = "\n".join(parts) if parts else "(no gene summary available)"

        return {
            "id":               f"ncbi_gene::{gene_id}",
            "title":            title,
            "url":              NCBIGeneTool._gene_url(gene_id),
            "text":             text,
            "source_type":      "ncbi_gene",
            "official_symbol":  symbol,
            "aliases":          aliases,
            "gene_description": desc,
        }

    # ------------------------------------------------------------------
    # Public tool method
    # ------------------------------------------------------------------

    def ncbi_gene_search(self, query: str, top_k: int = 5) -> list:
        """
        Search NCBI Gene for a human gene by symbol or name.

        Returns structured hits with the official gene symbol, known
        aliases/synonyms, and functional summary.  Use the exact gene
        symbol as the query (e.g. "IQGAP1") — do not include disease
        context in the ncbi_gene_search query.

        Returns: [{id, title, url, text, source_type, official_symbol,
                   aliases, gene_description}]
        """
        import threading
        _t0 = time.perf_counter()
        print(
            f"[NCBIGene] start | thread={threading.current_thread().name!r} | query={query!r}",
            flush=True,
        )

        gene_ids = self._search_gene_ids(query, retmax=top_k)
        print(
            f"[NCBIGene] esearch done | ids={gene_ids}"
            f" | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        if not gene_ids:
            print("[NCBIGene] no gene IDs — returning early", flush=True)
            return []

        raw_recs = self._fetch_gene_summaries(gene_ids[:top_k])
        print(
            f"[NCBIGene] esummary done | records={len(raw_recs)}"
            f" | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )

        hits = [self._format_hit(r) for r in raw_recs]
        print(
            f"[NCBIGene] done | hits={len(hits)}"
            f" | total_elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        return hits[:top_k]


if __name__ == "__main__":
    tool = NCBIGeneTool()

    test_genes = ["IQGAP1", "PIK3CD", "STK11", "BPGM", "ELOVL6"]
    print("=== ncbi_gene_search ===\n")
    for gene in test_genes:
        print(f"Query: {gene}")
        results = tool.ncbi_gene_search(gene, top_k=2)
        if results:
            for r in results:
                print(f"  [{r['id']}]")
                print(f"  Title   : {r['title'][:80]}")
                print(f"  URL     : {r['url']}")
                print(f"  Symbol  : {r['official_symbol']}  |  Aliases: {r['aliases'][:60]}")
                print(f"  Text[:200]: {r['text'][:200]}")
        else:
            print(f"  (no results for {gene})")
        print("-" * 50)
