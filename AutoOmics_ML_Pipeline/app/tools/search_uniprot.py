"""
search_uniprot.py — UniProt tool using the unipressed client.

unipressed wraps the official UniProt REST API v2 with typed queries,
auto-pagination, and field selection — analogous to how Biopython Entrez
wraps the NCBI API for PubMed.

No credentials required.
"""

from unipressed import UniprotkbClient


class UniProtTool:

    def _parse_entry(self, entry: dict, gene: str) -> dict | None:
        """Extract protein function + disease text from a UniProtKB entry."""
        accession = entry.get("primaryAccession", "")
        if not accession:
            return None

        # Gene symbol from entry (fall back to queried gene name)
        gene_list = entry.get("genes", [])
        symbol = (
            gene_list[0].get("geneName", {}).get("value", gene)
            if gene_list
            else gene
        )

        # Recommended protein name
        rec_name = (
            entry.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )

        # Walk comments for FUNCTION and DISEASE
        function_parts = []
        disease_parts = []
        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION":
                for t in comment.get("texts", []):
                    v = t.get("value", "").strip()
                    if v:
                        function_parts.append(v)
            elif ctype == "DISEASE":
                d_name = (
                    comment.get("disease", {})
                    .get("diseaseName", {})
                    .get("value", "")
                )
                texts = comment.get("texts", [])
                d_desc = texts[0].get("value", "").strip() if texts else ""
                entry_str = f"{d_name}: {d_desc}" if d_desc else d_name
                if entry_str.strip():
                    disease_parts.append(entry_str)

        parts = []
        if rec_name:
            parts.append(f"Protein: {rec_name}")
        if function_parts:
            parts.append("Function: " + " ".join(function_parts))
        if disease_parts:
            parts.append("Disease associations: " + "; ".join(disease_parts))

        text = "\n".join(parts).strip()
        if not text:
            return None

        return {
            "id": f"uniprot::{accession}",
            "title": f"{symbol} ({accession})",
            "url": f"https://www.uniprot.org/uniprotkb/{accession}",
            "text": text,
            "source_type": "uniprot",
        }

    def search_uniprot(self, gene: str, top_k: int = 3) -> list:
        """
        Search UniProtKB for a human gene (reviewed Swiss-Prot entries only).
        Returns protein function and disease association passages.

        Returns: [{id, title, url, text, source_type}]
        """
        query = f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true"
        try:
            search = UniprotkbClient.search(
                query=query,
                format="json",
                fields=["gene_names", "protein_name", "cc_function", "cc_disease"],
                size=top_k + 2,  # fetch a few extra in case some parse empty
            )
            entries = list(search.each_record())
        except Exception as e:
            print(f"[UniProt] search error for {gene}: {e}")
            return []

        results = []
        for entry in entries:
            parsed = self._parse_entry(entry, gene)
            if parsed:
                results.append(parsed)
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    tool = UniProtTool()
    for gene in ["CA1", "GYPA", "BPGM"]:
        print(f"\n=== {gene} ===")
        hits = tool.search_uniprot(gene, top_k=2)
        if not hits:
            print("  No results.")
        for r in hits:
            print(f"  {r['title']}")
            print(f"  {r['text'][:300]}")
            print()
