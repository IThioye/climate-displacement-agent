"""Small Langfuse v4 compatibility layer; observability never breaks the agent."""

from __future__ import annotations

import os
import threading
import logging
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The exporter otherwise prints multi-page connection tracebacks. The CLI reports
# one concise timeout message while LANGFUSE_DEBUG=true can restore SDK diagnostics.
if os.getenv("LANGFUSE_DEBUG", "false").lower() != "true":
    logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)

try:
    from langfuse import Langfuse, get_client, observe

    # Initialize once with conservative batching. Network export happens after the
    # answer instead of competing with an active tool or model call.
    _public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    _secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    _tracing_enabled = (
        bool(_public_key and _secret_key)
        and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "true"
    )
    _client = Langfuse(
        public_key=_public_key or None,
        secret_key=_secret_key or None,
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip("\"'"),
        tracing_enabled=_tracing_enabled,
        timeout=int(os.getenv("LANGFUSE_TIMEOUT_SECONDS", "5")),
        flush_at=int(os.getenv("LANGFUSE_FLUSH_AT", "1000")),
        flush_interval=float(os.getenv("LANGFUSE_FLUSH_INTERVAL_SECONDS", "60")),
    )

    LANGFUSE_AVAILABLE = True

    def update_current_span(**attributes) -> None:
        try:
            get_client().update_current_span(**attributes)
        except Exception:
            pass

    def update_current_generation(**attributes) -> None:
        try:
            get_client().update_current_generation(**attributes)
        except Exception:
            pass

    def current_trace_info() -> dict[str, str]:
        """Return local trace identifiers without making a network request."""
        try:
            client = get_client()
            trace_id = client.get_current_trace_id()
            if not trace_id:
                return {}
            info = {"trace_id": trace_id}
            project_id = os.getenv("LANGFUSE_PROJECT_ID", "").strip()
            if project_id:
                base_url = os.getenv(
                    "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
                ).strip("\"'").rstrip("/")
                info["trace_url"] = (
                    f"{base_url}/project/{project_id}/traces/{trace_id}"
                )
            return info
        except Exception:
            return {}

    def flush(timeout_seconds: float = 8.0) -> bool:
        """Try to export queued spans without letting telemetry hang the CLI."""
        completed = threading.Event()

        def send() -> None:
            try:
                get_client().flush()
            except Exception:
                pass
            finally:
                completed.set()

        threading.Thread(target=send, name="langfuse-flush", daemon=True).start()
        return completed.wait(timeout_seconds)

except ImportError:
    LANGFUSE_AVAILABLE = False

    def observe(*_args, **_kwargs):
        def decorator(function):
            return function
        return decorator

    def update_current_span(**_attributes) -> None:
        return None

    def update_current_generation(**_attributes) -> None:
        return None

    def current_trace_info() -> dict[str, str]:
        return {}

    def flush(timeout_seconds: float = 8.0) -> bool:
        del timeout_seconds
        return True
