"""Shared runtime result types for recoverable Agent failures."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class RuntimeFailure(BaseModel):
    status: Literal["incomplete", "error"]
    error_code: Literal["llm_timeout", "llm_unavailable", "tool_timeout", "invalid_output", "max_iterations", "internal_error"]
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class AgentRuntimeError(Exception):
    def __init__(self, failure: RuntimeFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


__all__ = ["RuntimeFailure", "AgentRuntimeError"]
