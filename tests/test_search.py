"""Tests for search tool."""

import pytest

from agent.tools.search import SearchTool


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_find_matching_file(self, tmp_path):
        (tmp_path / "hello.py").write_text("def greet():\n    return 'hello world'")
        (tmp_path / "other.py").write_text("import os\nprint('bye')")

        tool = SearchTool(root=tmp_path)
        result = await tool.run("hello")
        assert "hello.py" in result
        assert "Found" in result

    @pytest.mark.asyncio
    async def test_no_results(self, tmp_path):
        (tmp_path / "test.py").write_text("nothing here")
        tool = SearchTool(root=tmp_path)
        result = await tool.run("nonexistent_keyword_xyz")
        assert "No results" in result

    @pytest.mark.asyncio
    async def test_empty_query(self, tmp_path):
        tool = SearchTool(root=tmp_path)
        result = await tool.run("")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.py").write_text("Hello World")
        tool = SearchTool(root=tmp_path)
        result = await tool.run("hello")
        assert "test.py" in result

    @pytest.mark.asyncio
    async def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("password = 'secret'")
        (tmp_path / "visible.py").write_text("password = 'visible'")

        tool = SearchTool(root=tmp_path)
        result = await tool.run("password")
        assert "visible.py" in result
        assert ".hidden" not in result

    @pytest.mark.asyncio
    async def test_shows_line_numbers(self, tmp_path):
        (tmp_path / "test.py").write_text("line1\nline2\ntarget line\nline4")
        tool = SearchTool(root=tmp_path)
        result = await tool.run("target")
        assert "L3:" in result

    @pytest.mark.asyncio
    async def test_real_project_search(self):
        """Search the actual nano project for a known string."""
        from pathlib import Path
        tool = SearchTool(root=Path(__file__).parent.parent)
        result = await tool.run("OllamaClient")
        assert "ollama_client.py" in result
