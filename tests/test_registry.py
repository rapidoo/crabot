"""Tests for tool registry."""

import pytest

from agent.tools.base import Tool
from agent.tools.registry import register_tool, get_tool, list_tools, clear_registry


class DummyTool:
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool"

    async def run(self, input: str) -> str:
        return f"dummy:{input}"


class TestRegistry:
    def setup_method(self):
        clear_registry()

    def test_register_and_get(self):
        register_tool("dummy", lambda: DummyTool())
        tool = get_tool("dummy")
        assert tool is not None
        assert isinstance(tool, Tool)
        assert tool.name == "dummy"

    def test_get_unknown_returns_none(self):
        assert get_tool("nonexistent") is None

    def test_list_tools(self):
        register_tool("a", lambda: DummyTool())
        register_tool("b", lambda: DummyTool())
        names = list_tools()
        assert "a" in names
        assert "b" in names

    def test_factory_returns_none(self):
        register_tool("unavailable", lambda: None)
        assert get_tool("unavailable") is None

    def test_factory_called_lazily(self):
        calls = []

        def factory():
            calls.append(1)
            return DummyTool()

        register_tool("lazy", factory)
        assert len(calls) == 0
        get_tool("lazy")
        assert len(calls) == 1

    def test_clear_registry(self):
        register_tool("x", lambda: DummyTool())
        clear_registry()
        assert list_tools() == []
