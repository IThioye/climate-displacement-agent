# Homework requirement mapping

| Requirement | Status | Implementation evidence |
|---|---|---|
| BM25 + dense + RRF hybrid search | Implemented | `src/retrieval.py`: `BM25Index`, `DenseIndex`, `reciprocal_rank_fusion`, `ClimateRAG.search` |
| Cross-encoder before context assembly | Implemented | `Reranker.scores` ranks parent passages before `ClimateRAG.format_context` |
| Custom MCP server with 3+ tools | Implemented (4 tools) | `search_evidence`, `compare_regions`, `get_source`, `store_finding` in `src/mcp_server.py` |
| L1 input filtering | Implemented | Unicode normalization and injection patterns in `src/guardrails.py` |
| L4 action gating | Implemented | `ACTION_RISK_MATRIX`, default-confirm behavior, quotas and cost cap |
| Few-shot structured reasoning | Implemented | Two worked examples and required EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE headings in `src/reasoning.py` |
| Self-Consistency k=3 | Implemented | Three independently sampled candidates in `self_consistent_answer` |
| Langfuse LLM spans | Implemented | Every `ModelClient.complete` call is decorated as a generation span |
| Langfuse tool spans | Implemented | Flask/CLI `_search_evidence` and all four MCP tool functions have tool spans |
| Second agent role | Implemented | `CRITIC_PROMPT` checks citations, uncertainty and projection framing before return |

Langfuse instrumentation becomes visible in the selected Langfuse project when
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` are configured.
Without credentials the application still runs, but no remote trace can be shown.

