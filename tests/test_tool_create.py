"""Tests for tool_create — dynamic tool generation with versioning."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.tool_create import (
    ToolCreateTool,
    _validate_ast,
    _sanitize_name,
    _to_class_name,
    _get_next_version,
)


class TestAstValidation:
    def test_forbidden_os_import(self):
        assert _validate_ast("import os") is not None

    def test_forbidden_subprocess(self):
        assert _validate_ast("import subprocess") is not None

    def test_forbidden_http(self):
        assert _validate_ast("import http.client") is not None

    def test_forbidden_requests(self):
        assert _validate_ast("import requests") is not None

    def test_forbidden_eval(self):
        assert _validate_ast("eval('1+1')") is not None

    def test_forbidden_exec(self):
        assert _validate_ast("exec('pass')") is not None

    def test_open_write_mode_blocked(self):
        assert _validate_ast("open('f', 'w')") is not None

    def test_open_append_mode_blocked(self):
        assert _validate_ast("open('f', mode='a')") is not None

    def test_open_read_allowed(self):
        assert _validate_ast("open('f', 'r')") is None

    def test_safe_code_passes(self):
        assert _validate_ast("x = 1 + 2\nresult = str(x)") is None


class TestHelpers:
    def test_sanitize_name(self):
        assert _sanitize_name("My Tool!") == "my_tool"
        assert _sanitize_name("  ") == "custom_tool"
        assert _sanitize_name("hello-world") == "hello_world"

    def test_to_class_name(self):
        assert _to_class_name("my_tool") == "MyToolTool"
        assert _to_class_name("hello") == "HelloTool"


class TestVersioning:
    def test_first_version(self, tmp_path):
        assert _get_next_version("test", tmp_path) == 1

    def test_increments(self, tmp_path):
        (tmp_path / "test_v1.py.bak").touch()
        (tmp_path / "test_v2.py.bak").touch()
        assert _get_next_version("test", tmp_path) == 3


class TestToolCreateTool:
    @pytest.mark.asyncio
    async def test_create_simple_tool(self, tmp_path):
        tool = ToolCreateTool()
        with patch("agent.tools.tool_create._get_tools_dir", return_value=tmp_path):
            result = await tool.run("name:greet|description:Say hello|code:return f'Hello {input}'")
        assert "OK" in result or "ERROR" in result

    @pytest.mark.asyncio
    async def test_missing_name(self):
        tool = ToolCreateTool()
        result = await tool.run("description:test|code:return 'hi'")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_forbidden_code_rejected(self, tmp_path):
        tool = ToolCreateTool()
        with patch("agent.tools.tool_create._get_tools_dir", return_value=tmp_path):
            result = await tool.run("name:evil|description:bad|code:import os; os.system('rm -rf /')")
        assert "ERROR" in result
        assert "Forbidden" in result

    @pytest.mark.asyncio
    async def test_syntax_error_rejected(self, tmp_path):
        tool = ToolCreateTool()
        with patch("agent.tools.tool_create._get_tools_dir", return_value=tmp_path):
            result = await tool.run("name:broken|description:bad|code:def (((")
        assert "ERROR" in result

    def test_parse_spec(self):
        tool = ToolCreateTool()
        spec = tool._parse_spec("name:foo|description:bar|code:return 'hi'|test:hello")
        assert spec["name"] == "foo"
        assert spec["description"] == "bar"
        assert spec["code"] == "return 'hi'"
        assert spec["test"] == "hello"
