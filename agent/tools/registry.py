"""Tool registry — factory pattern with auto-discovery."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

_registry: dict[str, Callable[[], Tool | None]] = {}
_disabled_tools: set[str] = set()


def register_tool(name: str, factory: Callable[[], Tool | None]) -> None:
    """Register a tool factory. The factory returns None if the tool is unavailable."""
    _registry[name] = factory


def disable_tool(name: str) -> None:
    """Add a tool to the blocklist (used by evolution/action_applier)."""
    _disabled_tools.add(name)


def enable_tool(name: str) -> None:
    """Remove a tool from the blocklist."""
    _disabled_tools.discard(name)


def get_tool(name: str) -> Tool | None:
    """Retrieve a tool by name. Returns None if not registered, unavailable, or disabled."""
    if name in _disabled_tools:
        logger.info("Tool '%s' is disabled by evolution", name)
        return None
    factory = _registry.get(name)
    if factory is None:
        return None
    return factory()


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_registry.keys())


def list_tools_with_descriptions() -> list[dict[str, str]]:
    """Return tool names and descriptions for the planner prompt."""
    result: list[dict[str, str]] = []
    for name in _registry:
        tool = get_tool(name)
        if tool is not None:
            result.append({"name": name, "description": tool.description})
        else:
            result.append({"name": name, "description": "(unavailable)"})
    return result


def clear_registry() -> None:
    """Clear all registered tools (for testing)."""
    _registry.clear()


def discover_tools() -> None:
    """Auto-discover and import all tool modules in agent/tools/.

    Scans for .py files in the tools directory (and custom_tools/ if it exists).
    Each module is expected to call register_tool() at import time.
    """
    tools_dir = Path(__file__).parent
    _import_tools_from(tools_dir, "agent.tools")

    # Also scan custom_tools/ for user-generated tools
    custom_dir = tools_dir / "custom"
    if custom_dir.exists():
        _import_tools_from(custom_dir, "agent.tools.custom")


def _import_tools_from(directory: Path, package_prefix: str) -> None:
    """Import all .py modules from a directory."""
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name in ("base.py", "registry.py"):
            continue
        module_name = f"{package_prefix}.{py_file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.warning("Failed to load tool module %s: %s", module_name, exc)
