"""Regression checks for the code paths adapted directly from Labs B1-B4."""

from __future__ import annotations

import json

from src.agent import build_tool_registry
from llm_helpers import LLMClient, ToolRegistry, credentials_available
from src.guardrails import TokenBudget
from src.reasoning import _stance_signature


class DummyRAG:
    def search(self, query, k=4, region="", country=""):
        return [{"text": query, "metadata": {"region": region, "country": country}}]


def test_agent_uses_shared_lab_tool_registry():
    registry = build_tool_registry(DummyRAG())
    assert isinstance(registry, ToolRegistry)
    assert registry.names == ["search_evidence"]
    result = json.loads(registry.call(
        "search_evidence", {"query": "flood risk", "region": "Asia", "country": ""}
    ))
    assert result[0]["text"] == "flood risk"


def test_lab_b3_vote_groups_paraphrased_affirmative_conclusions():
    first = _stance_signature("Yes, the evidence shows increasing risk [1] [2].")
    second = _stance_signature("The reports indicate higher risk [2] and [1].")
    assert first == second


def test_local_ollama_has_zero_external_api_cost():
    budget = TokenBudget(max_usd=0.25)
    assert budget.record_tokens("ollama:gemma3:4b", 10_000, 2_000) == 0
    assert budget.spent == 0


def test_hosted_mistral_uses_compatible_endpoint_and_pricing(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    assert credentials_available("mistral")
    client = LLMClient(provider="mistral", model="mistral-small-latest")
    assert str(client._client.base_url).startswith("https://api.mistral.ai/v1")

    budget = TokenBudget(max_usd=1)
    cost = budget.record_tokens("mistral-small-latest", 1_000_000, 1_000_000)
    assert cost == 0.75
