import time

from dotenv import load_dotenv

load_dotenv()

from Bio import Entrez

from app.tools.entrez_utils import entrez_esearch, entrez_efetch

import numpy as np
from app.tools.tool_utils import chunk_text, get_model, encode_safe


class PubMedTool:
    def __init__(self):
        self.model = get_model()

    def search_pmids(self, query, retmax=10):
        """
        PubMed search → list of PMIDs (strings).
        """
        try:
            handle = entrez_esearch(
                db="pubmed",
                term=query,
                retmode="xml",
                retmax=retmax,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return list(record.get("IdList", []))
        except Exception as e:
            print(f"[PubMed] esearch error: {e}")
            return []

    def pubmed_url(self, pmid: str) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    def extract_abstract_text(self, art) -> str:
        """
        art = record["MedlineCitation"]["Article"]
        Handles list/string/missing AbstractText and section labels.
        """
        abs_node = art.get("Abstract")
        if not abs_node:
            return ""

        abs_text = abs_node.get("AbstractText")
        if not abs_text:
            return ""

        parts = []
        if isinstance(abs_text, (list, tuple)):
            for seg in abs_text:
                try:
                    text = str(seg).strip()
                    label = getattr(seg, "attributes", {}).get("Label")
                except Exception:
                    if isinstance(seg, str):
                        text = seg.strip()
                        label = None
                    elif isinstance(seg, dict):
                        text = (seg.get("_") or seg.get("__text") or "").strip()
                        label = seg.get("Label")
                    else:
                        text, label = "", None

                if text:
                    parts.append(f"{label}: {text}" if label else text)
        elif isinstance(abs_text, str):
            parts.append(abs_text.strip())

        return "\n".join(p for p in parts if p).strip()

    def get_title_and_abstract(self, pmid):
        """
        EFetch (XML) → extract ArticleTitle + AbstractText + Year.
        Returns None if no abstract.
        """
        try:
            handle = entrez_efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml",
            )
            record = Entrez.read(handle)
            handle.close()

            articles = record["PubmedArticle"]
            medline_citation = articles[0].get("MedlineCitation", {})
            art = medline_citation.get("Article", {})

            title = art.get("ArticleTitle", "")
            abstract_list = art.get("Abstract", {}).get("AbstractText")

            # Extract publication year
            year = None
            date_completed = medline_citation.get("DateCompleted", {})
            if date_completed:
                year = date_completed.get("Year")

            if not year:
                article_date = art.get("ArticleDate")
                if (
                    article_date
                    and isinstance(article_date, list)
                    and len(article_date) > 0
                ):
                    year = article_date[0].get("Year")

            if not year:
                journal = art.get("Journal", {})
                journal_issue = journal.get("JournalIssue", {})
                pub_date = journal_issue.get("PubDate", {})
                year = pub_date.get("Year")

            if not abstract_list:
                return {"title": str(title).strip(), "abstract": "", "year": year}

            parts = []
            for seg in abstract_list:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, dict):
                    # e.g., {"Label":"Background", "attributes":... , "__text":"..."}
                    text_val = seg.get("_", "") or seg.get("__text", "")
                    label = seg.get("Label")
                    if label and text_val:
                        parts.append(f"{label}: {text_val}")
                    elif text_val:
                        parts.append(text_val)

            abstract = " ".join([p.strip() for p in parts if p and p.strip()])
            return {
                "title": str(title).strip(),
                "abstract": abstract.strip(),
                "year": year,
            }

        except Exception as e:
            print(f"[PubMed] efetch error for PMID {pmid}: {e}")
            return None

    def pubmed_fetch_by_id(self, pmid: str) -> list:
        """
        Fetch the abstract for a specific known PubMed ID.
        Use this to hydrate a PMID cited inside UniProt function text or
        Open Targets evidence — returns the full abstract as a single result.

        Returns: [{id, title, url, year, text, source_type}]
        """
        meta = self.get_title_and_abstract(str(pmid))
        if not meta or not meta.get("abstract"):
            print(f"[PubMed] fetch_by_id: no abstract for PMID {pmid}")
            return []
        return [
            {
                "id": f"pubmed::{pmid}",
                "title": meta["title"],
                "url": self.pubmed_url(str(pmid)),
                "year": meta.get("year"),
                "text": meta["abstract"],
                "source_type": "pubmed",
            }
        ]

    def pubmed_semantic_search(self, query, top_k=5):
        """
        Search PubMed by query, fetch abstracts, chunk + encode, cosine-score to query.
        Returns: [{id,title,url,text,cos_sim_score,source_type}]
        """
        import threading
        _t0 = time.perf_counter()
        print(
            f"[PUBMED] start | thread={threading.current_thread().name!r} | query={query!r}",
            flush=True,
        )

        # --- Step 1: esearch ---
        print("[PUBMED] esearch start", flush=True)
        pmids = self.search_pmids(query, retmax=5)
        print(
            f"[PUBMED] esearch done  | pmids={pmids} | elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )

        if not pmids:
            print("[PUBMED] no PMIDs — returning early", flush=True)
            return []

        # --- Step 2: encode query ---
        print("[PUBMED] encode query start", flush=True)
        _te = time.perf_counter()
        q_vec = encode_safe(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        print(f"[PUBMED] encode query done | elapsed={time.perf_counter()-_te:.2f}s", flush=True)

        candidates = []
        for idx, pmid in enumerate(pmids):
            url = self.pubmed_url(pmid)

            # --- Step 3: efetch per PMID ---
            print(f"[PUBMED] efetch start | pmid={pmid} ({idx+1}/{len(pmids)})", flush=True)
            _tf = time.perf_counter()
            meta = self.get_title_and_abstract(pmid)
            print(
                f"[PUBMED] efetch done  | pmid={pmid} | elapsed={time.perf_counter()-_tf:.2f}s"
                f" | abstract={'yes' if meta and meta.get('abstract') else 'no/empty'}",
                flush=True,
            )

            if not meta:
                continue
            title = meta["title"]
            abstract = meta["abstract"]
            if not abstract.strip():
                continue

            chunks = chunk_text(abstract)

            # --- Step 4: encode chunks ---
            print(
                f"[PUBMED] encode chunks start | pmid={pmid} | chunks={len(chunks)}",
                flush=True,
            )
            _tc = time.perf_counter()
            ch_vecs = encode_safe(
                chunks, normalize_embeddings=True, convert_to_numpy=True
            )
            print(
                f"[PUBMED] encode chunks done  | pmid={pmid} | elapsed={time.perf_counter()-_tc:.2f}s",
                flush=True,
            )

            sims = np.dot(ch_vecs, q_vec)
            for ch, sim in zip(chunks, sims):
                candidates.append(
                    {
                        "id": f"pubmed::{pmid}",
                        "title": title,
                        "url": url,
                        "year": meta.get("year"),
                        "text": ch,
                        "cos_sim_score": float(sim),
                        "source_type": "pubmed",
                    }
                )

        candidates.sort(key=lambda d: d["cos_sim_score"], reverse=True)
        print(
            f"[PUBMED] done | candidates={len(candidates)} | total_elapsed={time.perf_counter()-_t0:.2f}s",
            flush=True,
        )
        return candidates[:top_k]


if __name__ == "__main__":
    pubmed_tool = PubMedTool()

    # --- pubmed_fetch_by_id ---
    test_pmids = [
        "30929741",   # SONFH / steroid-induced osteonecrosis
        "33761044",   # avascular necrosis coagulation
    ]
    print("=== pubmed_fetch_by_id ===")
    for pmid in test_pmids:
        results = pubmed_tool.pubmed_fetch_by_id(pmid)
        if results:
            r = results[0]
            print(f"PMID {pmid}: {r['title'][:80]}")
            print(f"  year={r['year']}  url={r['url']}")
            print(f"  abstract[:200]: {r['text'][:200]}")
        else:
            print(f"PMID {pmid}: no result")
        print("-" * 40)

    # --- pubmed_semantic_search ---
    query = [
        "Femoral Head Necrosis?",
        "How does anesthesia block pain receptors?",
        "Femoral Head Avascular Necrosis Joint Corticosteroids",
    ]

    print("\n=== pubmed_semantic_search ===")
    for q in query:
        print(f"Query: {q}")
        res = pubmed_tool.pubmed_semantic_search(q, top_k=5)
        for r in res:
            print(r)
        print("_" * 20)
