"""Agent memory — lightweight conversation and state management.

Plain dicts instead of LangChain message objects.
"""

from __future__ import annotations
from typing import Any


class AgentMemory:
    """Lightweight in-memory for agent conversation and intermediate state."""

    def __init__(self, messages: list[dict] | None = None) -> None:
        self.messages: list[dict] = list(messages or [])
        self.state: dict[str, Any] = {}

    def add_message(self, msg: dict) -> None:
        self.messages.append(msg)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_ai_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_chat_history(self, k: int = 20) -> list[dict]:
        return self.messages[-k:]

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def clear(self) -> None:
        self.messages.clear()
        self.state.clear()

    def snapshot(self) -> list[dict]:
        """Return a copy suitable for persistence."""
        return [dict(message) for message in self.messages]


__all__ = ["AgentMemory"]
