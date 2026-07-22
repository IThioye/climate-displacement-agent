"""Command-line climate-displacement evidence briefing agent."""

from __future__ import annotations

import argparse
import hashlib
import threading
import time
from typing import Callable

try:
    from langfuse.decorators import langfuse_context, observe
except ImportError:
    def observe(*_args, **_kwargs):
        def decorator(function):
            return function
        return decorator

    class _NoOpContext:
        def update_current_trace(self, **_kwargs):
            return None
    langfuse_context = _NoOpContext()

try:
    from .guardrails import TokenBudget, Verdict, l1_filter, l4_gate
    from .reasoning import SYSTEM_PROMPT, self_consistent_answer
    from .retrieval import ClimateRAG
except ImportError:
    from guardrails import TokenBudget, Verdict, l1_filter, l4_gate
    from reasoning import SYSTEM_PROMPT, self_consistent_answer
    from retrieval import ClimateRAG


AGENT_VERSION = "1.0.0"
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
ProgressCallback = Callable[..., None]
_RAG_INSTANCE: ClimateRAG | None = None
_RAG_LOCK = threading.Lock()


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    admin_message: str | None = None,
) -> None:
    if callback is not None:
        try:
            callback(stage, message, admin_message or message)
        except TypeError:
            callback(stage, message)
        except Exception:
            pass


def get_rag(event_callback: ProgressCallback | None = None) -> ClimateRAG:
    """Load the retrieval models/index once and reuse them across web requests."""
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _emit(
            event_callback, "retrieval",
            "Preparing the trusted document library for this session.",
            "Loading sanitized chunks, BM25 index, dense encoder, and cross-encoder reranker.",
        )
        with _RAG_LOCK:
            if _RAG_INSTANCE is None:
                _RAG_INSTANCE = ClimateRAG()
        _emit(
            event_callback,
            "retrieval",
            "The document library is ready.",
            f"Index ready: {len(_RAG_INSTANCE.chunks)} child chunks; "
            f"dense={_RAG_INSTANCE.dense.backend}; reranker={_RAG_INSTANCE.reranker.backend}.",
        )
    return _RAG_INSTANCE


@observe(name="tool_search_evidence")
def _search_evidence(
    rag: ClimateRAG,
    query: str,
    region: str,
    country: str,
) -> list[dict]:
    """Observed read-only tool boundary used by CLI and Flask agent runs."""
    return rag.search(query, k=4, region=region, country=country)


@observe(name="climate_displacement_agent")
def run(
    question: str,
    region: str = "",
    country: str = "",
    event_callback: ProgressCallback | None = None,
) -> dict:
    started = time.perf_counter()
    _emit(
        event_callback, "request",
        "Your question was received.",
        "Request received; applying Unicode normalization and L1 injection patterns.",
    )
    verdict, value = l1_filter(question, strict=True)
    if verdict is not Verdict.CLEAN:
        _emit(
            event_callback, "security",
            "The request was stopped by the safety check.",
            f"L1 blocked the request: {value}",
        )
        return {
            "answer": f"Request refused: {value}", "critic": "not run", "sources": [],
            "metrics": {"latency_s": round(time.perf_counter() - started, 3), "estimated_cost_usd": 0,
                        "tool_calls": {}, "agent_version": AGENT_VERSION, "prompt_hash": PROMPT_HASH},
            "outcome": "blocked",
        }

    _emit(
        event_callback, "security",
        "The request passed the safety check.",
        "L1 passed: input normalized and no injection pattern detected.",
    )

    budget = TokenBudget(max_usd=0.25)
    budget.record_tool_call("search_evidence")
    _emit(
        event_callback, "security",
        "Checking that the requested action is allowed.",
        "L4 checking search_evidence risk classification and per-run quota.",
    )
    allowed, reason = l4_gate("search_evidence", {"query": value, "region": region})
    if not allowed:
        _emit(event_callback, "security", "The requested action was not allowed.", f"L4 refused retrieval: {reason}")
        return {"answer": f"Retrieval refused: {reason}", "critic": "not run", "sources": [], "outcome": "blocked"}

    _emit(
        event_callback, "security",
        "The document search is authorized.",
        "L4 allowed the SAFE search_evidence tool; quota count recorded.",
    )
    rag = get_rag(event_callback)
    filters = ", ".join(value for value in (region, country) if value) or "none"
    _emit(
        event_callback, "retrieval",
        "Searching trusted reports for relevant evidence.",
        f"Executing BM25+dense retrieval and RRF fusion; metadata filters={filters}.",
    )
    evidence = _search_evidence(rag, value, region, country)
    if not evidence:
        _emit(
            event_callback, "retrieval",
            "No matching evidence was found in the available reports.",
            "No parent passages survived retrieval and metadata filtering.",
        )
        return {"answer": "No matching evidence was found in the local corpus.", "critic": "not run", "sources": [], "outcome": "no_evidence"}

    publishers = sorted({item["metadata"]["publisher"] for item in evidence})
    _emit(
        event_callback,
        "retrieval",
        "The most relevant report passages have been selected.",
        f"RRF fusion + cross-encoder selected {len(evidence)} parents from: {', '.join(publishers)}.",
    )

    context = rag.format_context(evidence)
    _emit(
        event_callback, "sanitization",
        "The selected report text passed a final content safety check.",
        "External-content sanitization reapplied immediately before context assembly.",
    )
    answer, critic = self_consistent_answer(
        value, context, budget, k=3, event_callback=event_callback
    )
    elapsed = time.perf_counter() - started
    sources = [
        {
            "citation": position,
            **{key: item["metadata"][key] for key in
               ("source_id", "title", "publisher", "year", "page", "url")},
        }
        for position, item in enumerate(evidence, start=1)
    ]
    langfuse_context.update_current_trace(
        name="climate-displacement-brief",
        version=AGENT_VERSION,
        metadata={"prompt_hash": PROMPT_HASH, "latency_s": elapsed, "cost_usd": budget.spent},
    )
    _emit(
        event_callback, "complete",
        "Your evidence briefing is ready.",
        f"Run completed in {elapsed:.2f}s with {len(sources)} cited passages.",
    )
    return {
        "answer": answer,
        "critic": critic,
        "sources": sources,
        "metrics": {
            "latency_s": round(elapsed, 3),
            "estimated_cost_usd": round(budget.spent, 6),
            "tool_calls": budget.tool_calls,
            "agent_version": AGENT_VERSION,
            "prompt_hash": PROMPT_HASH,
        },
        "outcome": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Question for the evidence agent")
    parser.add_argument("--region", default="")
    parser.add_argument("--country", default="")
    args = parser.parse_args()
    question = " ".join(args.question) or input("Question: ").strip()
    print("\nAI disclosure: this is an AI-generated research brief; verify cited sources.\n")
    result = run(question, region=args.region, country=args.country)
    print(result["answer"])
    print("\n--- CRITIC ---\n" + result["critic"])
    if result.get("sources"):
        print("\n--- SOURCES ---")
        for source in result["sources"]:
            print(f"- {source['publisher']} ({source['year']}), p.{source['page']}: {source['url']}")
    if result.get("metrics"):
        print("\n--- RUN METRICS ---")
        print(result["metrics"])


if __name__ == "__main__":
    main()
