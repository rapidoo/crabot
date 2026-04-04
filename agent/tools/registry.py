"""Tool registry — factory pattern for dynamic tool registration."""

from __future__ import annotations

from typing import Callable

from agent.tools.base import Tool

_registry: dict[str, Callable[[], Tool | None]] = {}


def register_tool(name: str, factory: Callable[[], Tool | None]) -> None:
    """Register a tool factory. The factory returns None if the tool is unavailable."""
    _registry[name] = factory


def get_tool(name: str) -> Tool | None:
    """Retrieve a tool by name. Returns None if not registered or unavailable."""
    factory = _registry.get(name)
    if factory is None:
        return None
    return factory()


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_registry.keys())


def clear_registry() -> None:
    """Clear all registered tools (for testing)."""
    _registry.clear()
