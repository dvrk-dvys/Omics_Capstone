import time
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
from Bio import Entrez

from app.tools.entrez_utils import entrez_esearch, entrez_efetch
from app.tools.tool_utils import chunk_text, encode_safe, get_model


class PMCTool:
    """
    PMC (PubMed Central) full-text article search via entrez_esearch + entrez_efetch.

    Abbreviation context:
      NIH  = National Institutes of Health (US government biomedical agency)
      NCBI = National Center for Biotechnology Information (division of NIH)
      PMC  = PubMed Central — NCBI's free full-text archive of biomedical literature
      Entrez = NCBI's unified API for querying databases (Gene, PubMed, GEO, PMC, etc.)

    Why this tool is different from pubmed_search:
      PubMed's esearch index covers only titles and abstracts.  A gene name or
      disease term that appears exclusively in the Methods, Results, or Discussion
      section of a paper will NOT be found by PubMed abstract search.  PMC
      efetch returns the full article XML body, enabling retrieval of mentions
      that standard PubMed queries miss.

    Use this tool in Iter 2 when:
      - Iter 1 pubmed_search returned weak or no disease-relevant results.
      - The gene is likely mentioned in article bodies (e.g. network/PPI studies,
        supplementary analyses, or short letters without dedicated abstracts).

    Full article body text is chunked, embedded, and ranked by cosine similarity
    to the query before returning top_k passages.

    Result source_type: "pmc"
    """

    def __init__(self):
        self.model = get_model()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pmc_url(pmcid: str) -> str:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"

    def _search_pmcids(self, query: str, retmax: int = 5) -> list:
        """PMC esearch → list of PMC ID strings (numeric digits only)."""
        try:
            handle = entrez_esearch(
                db="pmc",
                term=query,
                retmode="xml",
                retmax=retmax,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return list(record.get("IdList", []))
        except Exception as e:
            print(f"[PMC] esearch error: {e}")
            return []

    @staticmethod
    def _extract_text_from_xml(xml_bytes: bytes):
        """
        Parse PMC full-article XML, return (title: str, body_text: str).

        Walks <body> collecting text from all <title>, <p>, and <label>
        child elements.  Section headings are included for context continuity.
        """
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            print(f"[PMC] XML parse error: {e}")
            return "", ""

        title = ""
        for el in root.iter("article-title"):
            title = "".join(el.itertext()).strip()
            if title:
                break

        body_parts = []
        body_el = root.find(".//body")
        if body_el is not None:
            for child in body_el.iter():
                if child.tag in ("title", "p", "label"):
                    text = "".join(child.itertext()).strip()
                    if text:
                        body_parts.append(text)

        return title, "\n".join(body_parts)

    def _fetch_full_text(self, pmcid: str) -> Optional[dict]:
        """
        Efetch one PMC article (XML) → {pmcid, title, body_text}.
        Returns None on failure or empty body.
        """
        try:
            handle    = entrez_efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
            xml_bytes = handle.read()
            handle.close()

            if isinstance(xml_bytes, str):
                xml_bytes = xml_bytes.encode("utf-8")

            title, body_text = self._extract_text_from_xml(xml_bytes)
            if not body_text.strip():
                return None
            return {"pmcid": pmcid, "title": title, "body_text": body_text}
        except Exception as e:
            print(f"[PMC] efetch error for PMC{pmcid}: {e}")
            return None

    # ------------------------------------------------------------------
    # Public tool method
    # ------------------------------------------------------------------

    def pmc_fulltext_search(self, query: str, top_k: int = 5) -> list:
        """
        Search PMC full-text articles for query, embed body chunks, return the
        top_k most semantically similar passages.

        Use this when PubMed abstract evidence is weak, missing, or likely
        incomplete — PMC full-text recovers mentions that only appear in article
        bodies (Methods / Results / Discussion), not in PubMed-indexed abstracts.

        Returns: [{id, title, url, text, cos_sim_score, source_type="pmc"}]
        """
        import threading
        _t0 = time.perf_counter()
        print(
            f"[PMC] start | thread={threading.current_thread().name!r} | query={query!r}",
            flush=True,
        )

        pmcids = self._search_pmcids(query, retmax=5)
        print(
            f"[PMC] esearch done | pmcids={pmcids}"
            f" | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        if not pmcids:
            print("[PMC] no PMC IDs — returning early", flush=True)
            return []

        q_vec = encode_safe([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        candidates = []
        for idx, pmcid in enumerate(pmcids):
            print(f"[PMC] efetch start | PMC{pmcid} ({idx+1}/{len(pmcids)})", flush=True)
            _tf = time.perf_counter()
            article = self._fetch_full_text(pmcid)
            print(
                f"[PMC] efetch done | PMC{pmcid}"
                f" | elapsed={time.perf_counter()-_tf:.2f}s"
                f" | body={'yes' if article else 'no/empty'}",
                flush=True,
            )
            if not article:
                continue

            chunks = chunk_text(article["body_text"])
            if not chunks:
                continue

            ch_vecs = encode_safe(chunks, normalize_embeddings=True, convert_to_numpy=True)
            sims    = np.dot(ch_vecs, q_vec)
            for ch, sim in zip(chunks, sims):
                candidates.append({
                    "id":            f"pmc::{pmcid}",
                    "title":         article["title"],
                    "url":           self._pmc_url(pmcid),
                    "text":          ch,
                    "cos_sim_score": float(sim),
                    "source_type":   "pmc",
                })

        candidates.sort(key=lambda d: d["cos_sim_score"], reverse=True)
        print(
            f"[PMC] done | candidates={len(candidates)}"
            f" | total_elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        return candidates[:top_k]


if __name__ == "__main__":
    tool = PMCTool()

    test_queries = [
        "IQGAP1 osteonecrosis femoral head avascular necrosis",
        "STK11 bone ischemia steroid corticosteroid",
        "PIK3CD immune dysregulation osteonecrosis",
    ]
    print("=== pmc_fulltext_search ===\n")
    for q in test_queries:
        print(f"Query: {q}")
        results = tool.pmc_fulltext_search(q, top_k=3)
        if results:
            for r in results:
                pmcid = r["id"].split("::")[-1]
                print(f"  PMC{pmcid}  score={r['cos_sim_score']:.3f}")
                print(f"  Title: {r['title'][:80]}")
                print(f"  URL  : {r['url']}")
                print(f"  Text : {r['text'][:200]}")
        else:
            print("  (no results)")
        print("-" * 60)
