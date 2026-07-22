"""Climate adaptation of the course FastMCP server, with four guarded RAG tools."""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

try:
    from .guardrails import Verdict, l1_filter, l4_gate, sanitise_external_content
    from .observability import observe
    from .retrieval import ClimateRAG
except ImportError:
    from guardrails import Verdict, l1_filter, l4_gate, sanitise_external_content
    from observability import observe
    from retrieval import ClimateRAG


mcp = FastMCP("climate-displacement-evidence")
_rag: ClimateRAG | None = None
ROOT = Path(__file__).resolve().parents[1]


def get_rag() -> ClimateRAG:
    global _rag
    if _rag is None:
        _rag = ClimateRAG()
    return _rag


def safe_query(query: str) -> str:
    verdict, value = l1_filter(query, strict=True)
    if verdict is not Verdict.CLEAN:
        raise ValueError(value)
    return value


@mcp.tool()
@observe(name="tool_search_evidence", as_type="tool")
def search_evidence(query: str, region: str = "", country: str = "", k: int = 4) -> str:
    """Search the sanitized report corpus with hybrid retrieval and reranking.

    Use when: a claim needs supporting climate-displacement evidence.
    Do NOT use for: live news, personal data, or unsupported predictions.
    Returns: JSON evidence passages with publisher, year, page, URL and score.
    Example: search_evidence(query="flood displacement risk", country="Bangladesh")
    """
    try:
        query = safe_query(query)
        allowed, reason = l4_gate("search_evidence", {"query": query, "region": region})
        if not allowed:
            return json.dumps({"error": reason})
        results = get_rag().search(query, k=max(1, min(k, 8)), region=region, country=country)
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"search_evidence failed: {exc}"})


@mcp.tool()
@observe(name="tool_compare_regions", as_type="tool")
def compare_regions(query: str, regions: list[str], k_per_region: int = 2) -> str:
    """Retrieve comparable evidence separately for multiple regions.

    Use when: an analyst needs a source-balanced regional comparison.
    Do NOT use for: ranking people or making automated aid-allocation decisions.
    Returns: JSON mapping each region to its best sanitized evidence passages.
    Example: compare_regions(query="disaster displacement", regions=["South Asia", "Asia-Pacific"])
    """
    try:
        query = safe_query(query)
        if not regions or len(regions) > 5:
            raise ValueError("Provide between one and five regions")
        allowed, reason = l4_gate("compare_regions", {"query": query, "regions": regions})
        if not allowed:
            return json.dumps({"error": reason})
        result = {
            region: get_rag().search(query, k=max(1, min(k_per_region, 4)), region=region)
            for region in regions
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"compare_regions failed: {exc}"})


@mcp.tool()
@observe(name="tool_get_source", as_type="tool")
def get_source(source_id: str) -> str:
    """Return provenance metadata for one indexed report.

    Use when: verifying the origin, date, publisher, or URL of cited evidence.
    Do NOT use for: retrieving a report not listed in the local manifest.
    Returns: JSON source metadata or a structured not-found error.
    Example: get_source(source_id="idmc_grid_2025_summary")
    """
    try:
        source_id = safe_query(source_id)
        allowed, reason = l4_gate("get_source", {"source_id": source_id})
        if not allowed:
            return json.dumps({"error": reason})
        source = next((item for item in ClimateRAG.sources() if item["id"] == source_id), None)
        return json.dumps(source or {"error": "source not found"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"get_source failed: {exc}"})


@mcp.tool()
@observe(name="tool_store_finding", as_type="tool")
def store_finding(finding: str, source_id: str) -> str:
    """Store a human-verified finding in a local audit file.

    Use when: an analyst has manually verified a finding and its source.
    Do NOT use for: model speculation, uncited claims, or sensitive personal data.
    Returns: JSON confirmation; the action is recorded by the L4 monitor gate.
    Example: store_finding(finding="45.8m disaster displacements in 2024", source_id="idmc_grid_2025_summary")
    """
    try:
        finding, suspicious = sanitise_external_content(finding, max_chars=1_000)
        if suspicious:
            raise ValueError("finding contains instruction-like content")
        source_id = safe_query(source_id)
        allowed, reason = l4_gate("store_finding", {"finding": finding, "source_id": source_id})
        if not allowed:
            return json.dumps({"error": reason})
        target = ROOT / "data" / "processed" / "verified_findings.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"finding": finding, "source_id": source_id}) + "\n")
        return json.dumps({"stored": True, "source_id": source_id})
    except Exception as exc:
        return json.dumps({"error": f"store_finding failed: {exc}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
