"""Test configuration: unit tests must not export traces to the real project."""

import os


os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
