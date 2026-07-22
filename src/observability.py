"""Small Langfuse v4 compatibility layer; observability never breaks the agent."""

from __future__ import annotations

try:
    from langfuse import get_client, observe

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

    def flush() -> None:
        try:
            get_client().flush()
        except Exception:
            pass

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

    def flush() -> None:
        return None
