"""Tool interface — all tools implement this protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Interface every tool must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    async def run(self, input: str) -> str:
        """Execute the tool with the given input and return a string result."""
        ...
