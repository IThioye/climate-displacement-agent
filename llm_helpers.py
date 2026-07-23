"""Vendored homework subset of the course ``llm_helpers.py``.

It preserves the Lab B1-B4 interfaces used by this project: ``make_client``,
``AssistantMessage``, ``ToolRegistry``, and ``tool_schema``. The course-level file
is preferred when the original lab folder layout is present; this copy makes a
standalone clone runnable as well.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv


load_dotenv()
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
MISTRAL_OPENAI_BASE_URL = "https://api.mistral.ai/v1"


def credentials_available(provider: str | None = None) -> bool:
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "ollama":
        return True
    return bool(os.getenv({
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }.get(provider, "")))


@dataclass
class AssistantMessage:
    """Provider-independent response structure from the shared labs."""

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict | None = None

    def to_message(self) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """OpenAI-compatible portion of the course's normalized LLM client."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        from openai import OpenAI

        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", "gemma3:4b")
        if self.provider == "ollama":
            self._client = OpenAI(
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_OPENAI_BASE_URL),
            )
        elif self.provider == "google":
            self._client = OpenAI(
                api_key=os.environ["GOOGLE_API_KEY"], base_url=GOOGLE_OPENAI_BASE_URL
            )
        elif self.provider == "openai":
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL")
            )
        elif self.provider == "mistral":
            self._client = OpenAI(
                api_key=os.environ["MISTRAL_API_KEY"],
                base_url=os.getenv("MISTRAL_BASE_URL", MISTRAL_OPENAI_BASE_URL),
            )
        else:
            raise ValueError(f"Unsupported provider in vendored Lab helper: {self.provider!r}")

    def complete(
        self, messages: list[dict], tools: list[dict] | None = None,
        temperature: float = 0.0, max_tokens: int = 1024, tool_choice: Any = None,
    ) -> AssistantMessage:
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        calls = []
        for call in getattr(choice, "tool_calls", None) or []:
            arguments = call.function.arguments
            calls.append({
                "id": call.id,
                "name": call.function.name,
                "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
            })
        usage = response.usage
        return AssistantMessage(
            content=choice.content,
            tool_calls=calls,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        )


def make_client(
    offline_script: list | None = None, provider: str | None = None,
    model: str | None = None, force_mock: bool = False, quiet: bool = False,
):
    """Same factory contract used throughout the labs.

    The production agent requires a configured provider; its own safe extractive
    fallback handles an unavailable model. ``offline_script`` is accepted for API
    compatibility with the classroom helper.
    """
    del offline_script, force_mock
    selected = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if not credentials_available(selected):
        raise RuntimeError(f"No credentials configured for provider {selected!r}")
    if not quiet:
        print(f"Model provider: {selected}")
    return LLMClient(selected, model)


def tool_schema(
    name: str, description: str, properties: dict | None = None,
    required: list | None = None,
) -> dict:
    """Create the OpenAI-style tool schema used in Labs B1-B4."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties or {},
                "required": required or [],
            },
        },
    }


class ToolRegistry:
    """Associate Lab tool schemas with safe Python callables."""

    def __init__(self):
        self._specs: list[dict] = []
        self._funcs: dict[str, Callable[..., Any]] = {}

    def register(self, schema: dict, function: Callable[..., Any]) -> None:
        self._specs.append(schema)
        self._funcs[schema["function"]["name"]] = function

    @property
    def specs(self) -> list[dict]:
        return self._specs

    @property
    def names(self) -> list[str]:
        return [schema["function"]["name"] for schema in self._specs]

    def call(self, name: str, arguments: dict) -> str:
        if name not in self._funcs:
            return f"ERROR: unknown tool {name!r}"
        try:
            return str(self._funcs[name](**arguments))
        except Exception as exc:
            return f"ERROR while running {name!r}: {exc}"
