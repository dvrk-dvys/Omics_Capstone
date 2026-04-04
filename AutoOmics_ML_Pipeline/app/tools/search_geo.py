import time
from typing import Tuple

import numpy as np
from Bio import Entrez

from app.tools.entrez_utils import entrez_esearch, entrez_esummary
from app.tools.tool_utils import chunk_text, encode_safe, get_model


class GEOTool:
    """
    GEO (Gene Expression Omnibus) DataSets search via entrez_esearch + entrez_esummary.

    GEO  = Gene Expression Omnibus — NCBI's public repository of functional genomics data
           (microarray, RNA-seq, ChIP-seq, etc.)
    GDS  = GEO DataSets — the Entrez db name for GEO records (covers both GSE series
           and curated GDS accessions)

    This tool queries db="gds" to retrieve study-level metadata: title, study type,
    organism, platform technology, sample count, and a plain-language summary.
    It does NOT retrieve raw expression values — only dataset metadata.

    Use this tool in Iter 2 when dataset or study-level context could strengthen the
    interpretation, validate disease/source relevance, or ground expression findings
    in publicly available datasets.

    Result source_type: "geo"
    """

    def __init__(self):
        self.model = get_model()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geo_url(accession: str) -> str:
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"

    def _search_gds_uids(self, query: str, retmax: int = 10) -> list:
        """GEO DataSets esearch → list of GDS UID strings."""
        try:
            handle = entrez_esearch(
                db="gds",
                term=query,
                retmode="xml",
                retmax=retmax,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return list(record.get("IdList", []))
        except Exception as e:
            print(f"[GEO] esearch error: {e}")
            return []

    def _fetch_gds_summaries(self, uids: list) -> list:
        """GEO DataSets esummary for a list of UIDs → list of DocSum dicts."""
        if not uids:
            return []
        ids_str = ",".join(str(u) for u in uids)
        try:
            handle  = entrez_esummary(db="gds", id=ids_str)
            records = Entrez.read(handle)
            handle.close()
            if isinstance(records, list):
                return list(records)
            if hasattr(records, "get"):
                inner = records.get("DocumentSummarySet", {})
                if hasattr(inner, "get"):
                    return list(inner.get("DocumentSummary", []))
            return []
        except Exception as e:
            print(f"[GEO] esummary error: {e}")
            return []

    @staticmethod
    def _format_hit_text(rec, uid: str) -> Tuple[str, str, str]:
        """
        Extract (title, accession, text_blob) from a GDS DocSum record.
        All fields use .get() with empty-string fallbacks for resilience.
        """
        title     = str(rec.get("title",               "")).strip()
        accession = str(rec.get("Accession",           "")).strip()
        summary   = str(rec.get("summary",             "")).strip()
        taxon     = str(rec.get("taxon",               "")).strip()
        gds_type  = str(rec.get("gdsType",             "")).strip()
        n_samples = str(rec.get("n_samples",           "")).strip()
        platform  = str(rec.get("Platform_technology", "")).strip()
        organism  = str(rec.get("Platform_organism",   taxon)).strip()

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if accession:
            parts.append(f"Accession: {accession}")
        if gds_type:
            parts.append(f"Study type: {gds_type}")
        if organism:
            parts.append(f"Organism: {organism}")
        if n_samples:
            parts.append(f"Samples: {n_samples}")
        if platform:
            parts.append(f"Platform: {platform}")
        if summary:
            trunc = summary[:600] + ("..." if len(summary) > 600 else "")
            parts.append(f"Summary: {trunc}")

        text = "\n".join(parts) if parts else "(no GEO metadata available)"
        return (title or accession or uid), accession, text

    # ------------------------------------------------------------------
    # Public tool method
    # ------------------------------------------------------------------

    def geo_search(self, query: str, top_k: int = 5) -> list:
        """
        Search GEO DataSets for studies matching query, rank by semantic
        similarity, and return the top_k most relevant study records.

        Use this for dataset/study-level context expansion — helpful when
        literature alone is insufficient or when validating expression findings
        against publicly available datasets.

        Returns: [{id, title, url, text, cos_sim_score, source_type="geo", accession}]
        """
        import threading
        _t0 = time.perf_counter()
        print(
            f"[GEO] start | thread={threading.current_thread().name!r} | query={query!r}",
            flush=True,
        )

        uids = self._search_gds_uids(query, retmax=min(top_k * 2, 10))
        print(
            f"[GEO] esearch done | uids={uids}"
            f" | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        if not uids:
            print("[GEO] no GDS UIDs — returning early", flush=True)
            return []

        raw_recs = self._fetch_gds_summaries(uids)
        print(
            f"[GEO] esummary done | records={len(raw_recs)}"
            f" | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        if not raw_recs:
            return []

        q_vec = encode_safe([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        candidates = []
        for idx, rec in enumerate(raw_recs):
            uid = str(rec.get("Id", rec.get("uid", uids[idx] if idx < len(uids) else "?")))
            title, accession, text = self._format_hit_text(rec, uid)
            url = self._geo_url(accession) if accession else self._geo_url(uid)

            chunks = chunk_text(text)
            if not chunks:
                continue

            ch_vecs  = encode_safe(chunks, normalize_embeddings=True, convert_to_numpy=True)
            sims     = np.dot(ch_vecs, q_vec)
            best_sim = float(sims.max()) if len(sims) > 0 else 0.0

            # Return the full metadata blob (not just the best chunk) so the LLM
            # sees all study fields in one coherent hit.
            candidates.append({
                "id":            f"geo::{uid}",
                "title":         title,
                "url":           url,
                "text":          text,
                "cos_sim_score": best_sim,
                "source_type":   "geo",
                "accession":     accession,
            })

        candidates.sort(key=lambda d: d["cos_sim_score"], reverse=True)
        print(
            f"[GEO] done | candidates={len(candidates)}"
            f" | total_elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        return candidates[:top_k]


if __name__ == "__main__":
    tool = GEOTool()

    test_queries = [
        "osteonecrosis femoral head steroid peripheral blood microarray",
        "IQGAP1 expression profiling bone disease",
        "avascular necrosis corticosteroid Homo sapiens",
    ]
    print("=== geo_search ===\n")
    for q in test_queries:
        print(f"Query: {q}")
        results = tool.geo_search(q, top_k=3)
        if results:
            for r in results:
                print(f"  Accession: {r.get('accession', '?')}  score={r['cos_sim_score']:.3f}")
                print(f"  Title: {r['title'][:80]}")
                print(f"  URL  : {r['url']}")
                print(f"  Text : {r['text'][:200]}")
        else:
            print("  (no results)")
        print("-" * 60)
