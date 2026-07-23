"""Run the four required RAGAS metrics for baseline and final retrieval.

The Ollama path mirrors the repaired RAGAS cell in lab_B1_advanced_rag.ipynb.
A ten-question run makes many local judge calls and can take 15+ minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sys

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.retrieval import ClimateRAG


class LocalHashEmbeddings(Embeddings):
    """Deterministic fallback for providers such as Ollama without embedding APIs."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def judge_components():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    provider_defaults = {
        "ollama": "gemma3:4b",
        "mistral": "mistral-small-latest",
        "openai": "gpt-4o-mini",
        "google": "gemini-2.5-flash",
    }
    model = os.getenv("LLM_MODEL") or provider_defaults.get(provider, "gemma3:4b")
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        embeddings = LocalHashEmbeddings()
    elif provider == "google":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.environ["GOOGLE_API_KEY"]
        embeddings = LocalHashEmbeddings()
    elif provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
    elif provider == "mistral":
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        api_key = os.environ["MISTRAL_API_KEY"]
        embeddings = LocalHashEmbeddings()
    else:
        raise ValueError(f"RAGAS requires an OpenAI-compatible provider, got {provider!r}")
    judge = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url,
        temperature=0, max_retries=1, request_timeout=180,
    )
    return judge, embeddings, provider, model


def rows_for(cases: list[dict], rag: ClimateRAG, final: bool) -> dict:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for case in cases:
        results = rag.search(case["question"], k=4) if final else rag.baseline_search(case["question"], k=4)
        contexts = [result["text"] for result in results]
        rows["question"].append(case["question"])
        # Same extractive answer rule in both conditions isolates retrieval changes.
        rows["answer"].append(contexts[0] if contexts else "")
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(case["ground_truth"])
    return rows


def score(rows: dict, judge, embeddings) -> dict:
    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=180, max_retries=1, max_workers=1),
        raise_exceptions=True,
    )
    return {key: float(value) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Use fewer rows for a smoke test")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))[:args.limit]
    rag = ClimateRAG()
    judge, embeddings, provider, model = judge_components()
    print(f"RAGAS judge: provider={provider}, model={model}, questions={len(cases)}")
    output = {
        "provider": provider,
        "model": model,
        "questions": len(cases),
        "baseline": score(rows_for(cases, rag, final=False), judge, embeddings),
        "final": score(rows_for(cases, rag, final=True), judge, embeddings),
    }
    target = ROOT / "evaluation" / "ragas_results.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
