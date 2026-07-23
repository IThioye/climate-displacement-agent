"""Regression tests for local-model and tool-boundary failure handling."""

import numpy as np

from src import agent
from src.retrieval import DenseIndex, Reranker


class BrokenModel:
    def encode(self, *_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    def predict(self, *_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")


def test_dense_query_failure_rebuilds_compatible_hash_vectors():
    index = object.__new__(DenseIndex)
    index.texts = ["flood displacement Bangladesh", "drought Horn of Africa"]
    index.backend = "sentence-transformer (cuda)"
    index.model = BrokenModel()
    index.vectors = np.ones((2, 384), dtype=np.float32)

    scores = index.scores("Bangladesh flood")

    assert scores.shape == (2,)
    assert index.vectors.shape == (2, 512)
    assert index.backend == "hash-fallback"


def test_reranker_prediction_failure_uses_lexical_scores():
    reranker = object.__new__(Reranker)
    reranker.backend = "cross-encoder (cuda)"
    reranker.model = BrokenModel()

    scores = reranker.scores(
        "Bangladesh flood", ["Floods displaced people in Bangladesh."]
    )

    assert len(scores) == 1
    assert scores[0] > 0
    assert reranker.backend == "lexical-fallback"


def test_agent_returns_controlled_result_when_retrieval_tool_fails(monkeypatch):
    class FailedRag:
        chunks = []

        def search(self, *_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(agent, "get_rag", lambda _callback=None: FailedRag())

    result = agent.run("What does the evidence say about flood displacement?")

    assert result["outcome"] == "retrieval_error"
    assert result["critic"] == "not run"
    assert "CUDA out of memory" in result["answer"]
