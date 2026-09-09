"""Small, dependency-free JSONL request and runtime event logger."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_trace_context: ContextVar[dict[str, Any]] = ContextVar("trace_context", default={})


def configure_logging(path: str | Path) -> logging.Logger:
    logger = logging.getLogger("hr_agent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, "ts": time.time(), **_trace_context.get(), **fields}, ensure_ascii=False, default=str))


def current_trace_context() -> dict[str, Any]:
    return dict(_trace_context.get())


@contextmanager
def trace_context(**fields: Any):
    """Attach request-scoped identifiers to all nested runtime events."""
    current = dict(_trace_context.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _trace_context.set(current)
    try:
        yield
    finally:
        _trace_context.reset(token)


__all__ = ["configure_logging", "log_event", "trace_context", "current_trace_context"]
