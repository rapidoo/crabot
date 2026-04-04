"""Tests for file_io and code_exec tools."""

import pytest

from agent.tools.file_io import FileIOTool
from agent.tools.code_exec import CodeExecTool


class TestFileIOTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")

        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run(f"read:{f}")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run(f"read:{tmp_path}/nope.txt")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_read_outside_root(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run("read:/etc/passwd")
        assert "ERROR" in result
        assert "outside" in result

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run(f"write:{tmp_path}/out.txt:hello")
        assert "OK" in result
        assert (tmp_path / "out.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run(f"write:{tmp_path}/sub/dir/out.txt:data")
        assert "OK" in result
        assert (tmp_path / "sub" / "dir" / "out.txt").exists()

    @pytest.mark.asyncio
    async def test_write_outside_root(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run("write:/tmp/evil.txt:hack")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tmp_path):
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run("delete:foo")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_read_relative_path(self, tmp_path):
        (tmp_path / "rel.txt").write_text("relative")
        tool = FileIOTool(allowed_root=tmp_path)
        result = await tool.run("read:rel.txt")
        assert result == "relative"


class TestCodeExecTool:
    @pytest.mark.asyncio
    async def test_simple_print(self):
        tool = CodeExecTool()
        result = await tool.run("print(1 + 1)")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_multiline(self):
        code = "for i in range(3):\n    print(i)"
        tool = CodeExecTool()
        result = await tool.run(code)
        assert "0" in result
        assert "1" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_error_returns_stderr(self):
        tool = CodeExecTool()
        result = await tool.run("raise ValueError('boom')")
        assert "ValueError" in result
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = CodeExecTool(timeout=2)
        result = await tool.run("import time; time.sleep(10)")
        assert "TIMEOUT" in result

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        tool = CodeExecTool()
        result = await tool.run("def :")
        assert "SyntaxError" in result

    @pytest.mark.asyncio
    async def test_captures_stdout(self):
        tool = CodeExecTool()
        result = await tool.run("import sys; sys.stdout.write('out')")
        assert "out" in result
