"""Command-line climate-displacement evidence briefing agent."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable
import warnings

warnings.filterwarnings("ignore")

# Reuse Lab B1/B2's ToolRegistry and tool-schema contract for in-process tool
# execution. The same domain operations are separately exposed over MCP.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if str(COURSE_ROOT) not in sys.path:
    sys.path.append(str(COURSE_ROOT))
from llm_helpers import ToolRegistry, tool_schema

try:
    from .guardrails import TokenBudget, Verdict, l1_filter, l4_gate
    from .observability import current_trace_info, flush as flush_langfuse
    from .observability import observe, update_current_span
    from .reasoning import SYSTEM_PROMPT, self_consistent_answer
    from .retrieval import ClimateRAG
except ImportError:
    from guardrails import TokenBudget, Verdict, l1_filter, l4_gate
    from observability import current_trace_info, flush as flush_langfuse
    from observability import observe, update_current_span
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


@observe(
    name="tool_search_evidence",
    as_type="tool",
    capture_input=False,
    capture_output=False,
)
def _search_evidence(
    rag: ClimateRAG,
    query: str,
    region: str,
    country: str,
    event_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Observed read-only tool boundary used by CLI and Flask agent runs.

    The RAG object is deliberately excluded from automatic span serialization:
    it contains models, vectors, indexes, and the complete document corpus.
    """
    update_current_span(
        input={"query": query, "region": region, "country": country},
        metadata={"tool": "search_evidence"},
    )

    def retrieval_progress(message: str) -> None:
        _emit(event_callback, "retrieval", message, message)

    search_arguments = {"k": 4, "region": region, "country": country}
    if "event_callback" in inspect.signature(rag.search).parameters:
        search_arguments["event_callback"] = retrieval_progress
    results = rag.search(query, **search_arguments)
    update_current_span(
        output={"result_count": len(results)},
        metadata={"tool": "search_evidence", "result_count": len(results)},
    )
    return results


def build_tool_registry(
    rag: ClimateRAG,
    event_callback: ProgressCallback | None = None,
) -> ToolRegistry:
    """Adapt the shared Lab registry to the climate-displacement search tool."""
    registry = ToolRegistry()

    def search_evidence(query: str, region: str = "", country: str = "") -> str:
        return json.dumps(
            _search_evidence(rag, query, region, country, event_callback),
            ensure_ascii=False,
        )

    registry.register(
        tool_schema(
            "search_evidence",
            "Search approved climate-displacement reports before answering factual questions.",
            {
                "query": {"type": "string", "description": "Evidence question"},
                "region": {"type": "string", "description": "Optional region"},
                "country": {"type": "string", "description": "Optional country"},
            },
            ["query"],
        ),
        search_evidence,
    )
    return registry


@observe(name="climate_displacement_agent", as_type="agent")
def run(
    question: str,
    region: str = "",
    country: str = "",
    event_callback: ProgressCallback | None = None,
) -> dict:
    started = time.perf_counter()
    update_current_span(
        version=AGENT_VERSION,
        metadata={"prompt_hash": PROMPT_HASH, "component": "agent"},
    )
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
    registry = build_tool_registry(rag, event_callback)
    raw_evidence = registry.call(
        "search_evidence", {"query": value, "region": region, "country": country}
    )
    try:
        evidence = json.loads(raw_evidence)
    except (json.JSONDecodeError, TypeError):
        detail = str(raw_evidence)
        if detail.startswith("ERROR while running"):
            detail = detail.split(":", 1)[-1].strip()
        _emit(
            event_callback, "error",
            "The document search could not be completed.",
            f"search_evidence failed: {detail}",
        )
        elapsed = time.perf_counter() - started
        return {
            "answer": (
                "I could not search the document library, so I cannot produce a "
                f"grounded answer. Retrieval error: {detail}"
            ),
            "critic": "not run",
            "sources": [],
            "metrics": {
                "latency_s": round(elapsed, 3),
                "estimated_cost_usd": round(budget.spent, 6),
                "tool_calls": budget.tool_calls,
                "agent_version": AGENT_VERSION,
                "prompt_hash": PROMPT_HASH,
            },
            "outcome": "retrieval_error",
        }
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
    update_current_span(
        version=AGENT_VERSION,
        metadata={"prompt_hash": PROMPT_HASH, "latency_s": elapsed, "cost_usd": budget.spent},
    )
    trace = current_trace_info()
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
            **trace,
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

    stage_labels = {
        "request": "Request",
        "security": "Safety",
        "retrieval": "Evidence",
        "sanitization": "Sanitization",
        "reasoning": "Analysis",
        "critic": "Review",
        "complete": "Complete",
        "error": "Error",
    }

    def show_progress(stage: str, message: str, _admin_message: str = "") -> None:
        """Print the same user-friendly live stages shown by the Flask interface."""
        label = stage_labels.get(stage, stage.replace("_", " ").title())
        print(f"[{label}] {message}", flush=True)

    print("\nAI disclosure: this is an AI-generated research brief; verify cited sources.\n")
    result = run(
        question,
        region=args.region,
        country=args.country,
        event_callback=show_progress,
    )
    print()
    print(result["answer"])
    print("\n--- CRITIC ---\n" + result["critic"])
    if result.get("sources"):
        print("\n--- SOURCES ---")
        for source in result["sources"]:
            print(f"- {source['publisher']} ({source['year']}), p.{source['page']}: {source['url']}")
    if result.get("metrics"):
        print("\n--- RUN METRICS ---")
        print(result["metrics"])
    trace_url = result.get("metrics", {}).get("trace_url")
    trace_id = result.get("metrics", {}).get("trace_id")
    if trace_url:
        print(f"\n--- LANGFUSE TRACE ---\n{trace_url}")
    elif trace_id:
        print(
            "\n--- LANGFUSE TRACE ---\n"
            f"Trace ID: {trace_id}\n"
            "Open your Langfuse project and search for this trace ID."
        )
    elif result.get("metrics"):
        print("\n--- LANGFUSE TRACE ---\nNo trace ID was created; check the Langfuse keys.")

    if not flush_langfuse():
        print(
            "Trace export did not finish within 8 seconds. The answer is complete; "
            "check LANGFUSE_BASE_URL, connectivity, or proxy settings."
        )


if __name__ == "__main__":
    main()
