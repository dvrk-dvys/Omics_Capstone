"""
search_opentargets.py — Open Targets Platform tool via the official GraphQL API.

Open Targets does not maintain a Python client; the official access model is
direct GraphQL queries to their endpoint. We use `gql` as a structured
GraphQL client, consistent with how unipressed wraps UniProt.

No credentials required.
Endpoint: https://api.platform.opentargets.org/api/v4/graphql
"""

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport


OT_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

_SEARCH_QUERY = gql("""
query Search($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits {
      id
      name
      entity
    }
  }
}
""")

_TARGET_QUERY = gql("""
query Target($id: String!) {
  target(ensemblId: $id) {
    approvedSymbol
    approvedName
    functionDescriptions
    associatedDiseases(page: {index: 0, size: 8}) {
      rows {
        disease {
          id
          name
          description
        }
        score
      }
    }
  }
}
""")


class OpenTargetsTool:

    def _make_client(self):
        """Create a fresh gql Client per call — required for thread safety."""
        transport = RequestsHTTPTransport(url=OT_ENDPOINT, timeout=15)
        return Client(transport=transport, fetch_schema_from_transport=False)

    def _resolve_target_id(self, gene: str, client) -> tuple[str, str] | tuple[None, None]:
        """Map gene symbol → (ensembl_id, approved_name). Returns (None, None) on miss."""
        try:
            res = client.execute(_SEARCH_QUERY, variable_values={"q": gene})
            hits = res.get("search", {}).get("hits", [])
            if not hits:
                return None, None
            return hits[0]["id"], hits[0]["name"]
        except Exception as e:
            print(f"[OpenTargets] search error for {gene}: {e}")
            return None, None

    def search_opentargets(self, gene: str, top_k: int = 5) -> list:
        """
        Query Open Targets for a gene's function description and top disease
        associations. Returns passages in the same format as other tools.

        Returns: [{id, title, url, text, source_type}]
        """
        client = self._make_client()
        ensembl_id, approved_name = self._resolve_target_id(gene, client)
        if not ensembl_id:
            print(f"[OpenTargets] no target found for {gene}")
            return []

        target_url = f"https://platform.opentargets.org/target/{ensembl_id}"

        try:
            res = client.execute(_TARGET_QUERY, variable_values={"id": ensembl_id})
        except Exception as e:
            print(f"[OpenTargets] target query error for {gene}: {e}")
            return []

        target = res.get("target", {})
        if not target:
            return []

        results = []

        # --- Function description entry ---
        func_descs = target.get("functionDescriptions", [])
        if func_descs:
            # Strip trailing evidence codes {ECO:...} for readability
            func_text = func_descs[0].split("{ECO:")[0].strip()
            results.append({
                "id": f"opentargets::{ensembl_id}::function",
                "title": f"{gene} — {approved_name or target.get('approvedName', '')}",
                "url": target_url,
                "text": f"Function: {func_text}",
                "source_type": "opentargets",
            })

        # --- Disease association entries ---
        rows = target.get("associatedDiseases", {}).get("rows", [])
        for row in rows:
            disease = row.get("disease", {})
            d_name = disease.get("name", "")
            d_desc = disease.get("description", "")
            score = row.get("score", 0.0)

            if not d_name:
                continue

            text_parts = [f"Disease: {d_name}"]
            if d_desc:
                text_parts.append(f"Description: {d_desc}")
            text_parts.append(f"Association score: {score:.3f}")

            results.append({
                "id": f"opentargets::{ensembl_id}::{disease.get('id', d_name)}",
                "title": f"{gene} ↔ {d_name}",
                "url": f"{target_url}/associations",
                "text": "\n".join(text_parts),
                "cos_sim_score": float(score),
                "source_type": "opentargets",
            })

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    tool = OpenTargetsTool()
    for gene in ["CA1", "GYPA", "BPGM"]:
        print(f"\n=== {gene} ===")
        hits = tool.search_opentargets(gene, top_k=4)
        if not hits:
            print("  No results.")
        for r in hits:
            print(f"  {r['title']}")
            print(f"  {r['text'][:250]}")
            print()
