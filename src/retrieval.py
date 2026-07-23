"""Parent-child hybrid RAG: BM25 + dense retrieval + RRF + reranking.

The architecture is adapted from lab_B1_advanced_rag.ipynb. Real
sentence-transformer models are preferred; deterministic local fallbacks keep a
fresh clone runnable when model downloads are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from .guardrails import sanitise_external_content
    from .ingest import CACHE_PATH, MANIFEST_PATH, build_chunks
except ImportError:
    from guardrails import sanitise_external_content
    from ingest import CACHE_PATH, MANIFEST_PATH, build_chunks


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Small BM25-Okapi implementation from the advanced RAG lab."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [tokenise(text) for text in texts]
        count = max(1, len(texts))
        self.avgdl = sum(map(len, self.tokens)) / count
        document_frequency = Counter(word for words in self.tokens for word in set(words))
        self.idf = {
            word: math.log((count - frequency + 0.5) / (frequency + 0.5) + 1)
            for word, frequency in document_frequency.items()
        }
        self.k1, self.b = k1, b

    def score(self, query: str, index: int) -> float:
        query_tokens = tokenise(query)
        document = self.tokens[index]
        frequencies = Counter(document)
        score = 0.0
        for word in query_tokens:
            frequency = frequencies[word]
            if not frequency or word not in self.idf:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * len(document) / max(self.avgdl, 1)
            )
            score += self.idf[word] * frequency * (self.k1 + 1) / denominator
        return score


class TfidfIndex:
    """Basic TF-IDF cosine baseline matching the first stage of Lab B1."""

    def __init__(self, texts: list[str]):
        tokens = [tokenise(text) for text in texts]
        count = max(1, len(tokens))
        document_frequency = Counter(word for words in tokens for word in set(words))
        self.idf = {
            word: math.log((1 + count) / (1 + frequency)) + 1
            for word, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(words) for words in tokens]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        frequencies = Counter(tokens)
        length = len(tokens) or 1
        return {
            word: frequency / length * self.idf.get(word, 0.0)
            for word, frequency in frequencies.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(left[word] * right[word] for word in set(left) & set(right))
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def scores(self, query: str) -> list[float]:
        query_vector = self._vector(tokenise(query))
        return [self._cosine(query_vector, vector) for vector in self.vectors]


class DenseIndex:
    """Sentence-transformer embeddings with a no-network hash fallback."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.backend = "hash-fallback"
        self.model = None
        use_models = os.getenv("RAG_USE_LOCAL_MODELS", "1") == "1"
        if use_models:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(os.getenv(
                    "DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ), device=os.getenv("RAG_DEVICE", "cpu"))
                self.backend = f"sentence-transformer ({os.getenv('RAG_DEVICE', 'cpu')})"
            except Exception as exc:
                print(f"[WARN] Dense model unavailable; using hash fallback: {exc}")
        try:
            self.vectors = self.encode(self.texts)
        except Exception as exc:
            self._switch_to_fallback(exc)

    def _switch_to_fallback(self, exc: Exception) -> None:
        """Release a failed accelerator/model and rebuild compatible hash vectors."""
        print(f"[WARN] Dense encoding failed; using hash fallback: {exc}")
        self.model = None
        self.backend = "hash-fallback"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        self.vectors = np.vstack(
            [self._hash_vector(text) for text in self.texts]
        ) if self.texts else np.empty((0, 512))

    @staticmethod
    def _hash_vector(text: str, dimensions: int = 512) -> np.ndarray:
        vector = np.zeros(dimensions, dtype=np.float32)
        for token in tokenise(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            vector[index] += 1 if digest[4] & 1 else -1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return np.vstack([self._hash_vector(text) for text in texts]) if texts else np.empty((0, 512))

    def scores(self, query: str) -> np.ndarray:
        try:
            query_vector = self.encode([query])[0]
        except Exception as exc:
            self._switch_to_fallback(exc)
            query_vector = self._hash_vector(query)
        return self.vectors @ query_vector


class Reranker:
    """Cross-encoder reranker with the lab's lexical fallback."""

    def __init__(self):
        self.model = None
        self.backend = "lexical-fallback"
        if os.getenv("RAG_USE_LOCAL_MODELS", "1") == "1":
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(os.getenv(
                    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ), device=os.getenv("RAG_DEVICE", "cpu"))
                self.backend = f"cross-encoder ({os.getenv('RAG_DEVICE', 'cpu')})"
            except Exception as exc:
                print(f"[WARN] Cross-encoder unavailable; using lexical fallback: {exc}")

    @staticmethod
    def _fallback_score(query: str, document: str) -> float:
        query_tokens, document_tokens = set(tokenise(query)), set(tokenise(document))
        if not query_tokens or not document_tokens:
            return 0.0
        coverage = len(query_tokens & document_tokens) / len(query_tokens)
        return 0.8 * coverage + 0.2 * min(1.0, len(document_tokens) / 100)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if self.model is not None:
            try:
                return [
                    float(value)
                    for value in self.model.predict([(query, doc) for doc in documents])
                ]
            except Exception as exc:
                print(f"[WARN] Cross-encoder prediction failed; using lexical fallback: {exc}")
                self.model = None
                self.backend = "lexical-fallback"
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
        return [self._fallback_score(query, document) for document in documents]


def reciprocal_rank_fusion(rankings: list[list[int]], constant: int = 60) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (constant + rank)
    return [index for index, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]


class ClimateRAG:
    """Search sanitized climate-displacement evidence with source metadata."""

    def __init__(self, cache_path: Path = CACHE_PATH):
        if not cache_path.exists():
            build_chunks(output_path=cache_path)
        self.chunks = json.loads(cache_path.read_text(encoding="utf-8"))
        if not self.chunks:
            raise RuntimeError("No document chunks found. Download documents and run src/ingest.py")
        self.texts = [chunk["text"] for chunk in self.chunks]
        self.tfidf = TfidfIndex(self.texts)
        self.bm25 = BM25Index(self.texts)
        self.dense = DenseIndex(self.texts)
        self.reranker = Reranker()

    @staticmethod
    def _matches(metadata: dict, filters: dict) -> bool:
        for key, wanted in filters.items():
            if wanted in (None, "", 0):
                continue
            actual = metadata.get(key)
            if isinstance(wanted, str):
                if wanted.lower() not in str(actual).lower():
                    return False
            elif actual != wanted:
                return False
        return True

    def _eligible(self, filters: dict) -> list[int]:
        return [
            index for index, chunk in enumerate(self.chunks)
            if self._matches(chunk["metadata"], filters)
        ]

    def search(
        self,
        query: str,
        k: int = 4,
        candidate_k: int = 30,
        event_callback: Callable[[str], None] | None = None,
        **filters,
    ) -> list[dict]:
        def progress(message: str) -> None:
            if event_callback is not None:
                event_callback(message)

        eligible = self._eligible(filters)
        if not eligible:
            return []
        progress("Ranking matching passages with keyword and semantic search.")
        bm25_ranking = sorted(eligible, key=lambda i: self.bm25.score(query, i), reverse=True)
        dense_scores = self.dense.scores(query)
        dense_ranking = sorted(eligible, key=lambda i: float(dense_scores[i]), reverse=True)
        progress("Combining the keyword and semantic rankings.")
        candidates = reciprocal_rank_fusion([
            bm25_ranking[:candidate_k], dense_ranking[:candidate_k]
        ])[:candidate_k]

        # Retrieve small child chunks, but rerank/return their richer parents.
        parents: dict[str, dict] = {}
        for index in candidates:
            chunk = self.chunks[index]
            parents.setdefault(chunk["parent_id"], chunk)
        parent_chunks = list(parents.values())
        parent_texts = [chunk["parent_text"] for chunk in parent_chunks]
        progress("Reranking the strongest passages before assembling evidence.")
        rerank_scores = self.reranker.scores(query, parent_texts)
        ranked = sorted(
            zip(rerank_scores, parent_chunks), key=lambda item: item[0], reverse=True
        )[:k]
        results = []
        for score, chunk in ranked:
            text, suspicious = sanitise_external_content(chunk["parent_text"])
            results.append({
                "score": round(float(score), 5),
                "text": text,
                "metadata": chunk["metadata"],
                "security_flag": suspicious,
            })
        return results

    def baseline_search(self, query: str, k: int = 4, **filters) -> list[dict]:
        """Basic TF-IDF baseline for the before/after evaluation."""
        eligible = self._eligible(filters)
        scores = self.tfidf.scores(query)
        ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"score": scores[index], "text": self.chunks[index]["text"],
             "metadata": self.chunks[index]["metadata"]}
            for index in ranked
        ]

    @staticmethod
    def format_context(results: list[dict]) -> str:
        blocks = []
        for position, result in enumerate(results, start=1):
            meta = result["metadata"]
            blocks.append(
                f"[{position}] {meta['publisher']} ({meta['year']}), {meta['title']}, "
                f"page {meta['page']}\n{result['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def sources() -> list[dict]:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sources"]
