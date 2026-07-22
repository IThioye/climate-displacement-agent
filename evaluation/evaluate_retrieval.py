"""Reproducible BM25 baseline versus final hybrid retrieval comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import ClimateRAG


def source_rank(results: list[dict], source_id: str) -> int:
    for rank, result in enumerate(results, start=1):
        if result["metadata"]["source_id"] == source_id:
            return rank
    return 0


def summarize(ranks: list[int]) -> dict:
    return {
        "hit@4": round(sum(rank > 0 for rank in ranks) / len(ranks), 3),
        "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 3),
    }


def main() -> None:
    cases = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    rag = ClimateRAG()
    baseline, final = [], []
    for case in cases:
        baseline.append(source_rank(rag.baseline_search(case["question"], k=4), case["source_id"]))
        final.append(source_rank(rag.search(case["question"], k=4), case["source_id"]))
    print(json.dumps({"baseline": summarize(baseline), "final": summarize(final)}, indent=2))
    print("Use the same questions in the RAGAS notebook/script for the four required generation metrics.")


if __name__ == "__main__":
    main()
