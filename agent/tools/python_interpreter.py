"""Stateful Python interpreter tool — executes code with persistent namespace."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import functools
import io
import traceback
from typing import Any

from agent.tools.registry import register_tool

DEFAULT_TIMEOUT = 30

# Modules the interpreter is allowed to import
_SAFE_MODULES = frozenset({
    "json", "math", "re", "collections", "datetime", "statistics",
    "itertools", "functools", "textwrap", "string", "decimal",
    "fractions", "random", "hashlib", "base64", "csv", "io",
    "dataclasses", "enum", "typing", "copy", "operator",
    "pathlib",
})

# Builtins that are blocked
_BLOCKED_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "breakpoint",
    "exit", "quit", "open",
})


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Import guard — only allow whitelisted modules."""
    top = name.split(".")[0]
    if top not in _SAFE_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __builtins_original_import__(name, *args, **kwargs)


# Keep a reference to the real __import__
__builtins_original_import__ = builtins.__import__


def _make_safe_builtins() -> dict[str, Any]:
    """Build a restricted builtins dict."""
    safe = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS}
    safe["__import__"] = _safe_import
    return safe


class PythonInterpreterTool:
    """Execute Python code in a stateful in-process interpreter.

    Variables, functions, and (whitelisted) imports persist across calls.
    Dangerous operations are blocked.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._timeout = timeout
        self._safe_builtins = _make_safe_builtins()
        self._namespace: dict[str, Any] = {"__builtins__": self._safe_builtins}

    @property
    def name(self) -> str:
        return "python_interpreter"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a stateful interpreter. "
            "Variables and imports persist across calls. "
            "Print output to return results."
        )

    async def run(self, input: str) -> str:
        """Execute code and return captured stdout, or an error message."""
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, functools.partial(self._exec, input)),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            return "ERROR: execution timed out"

    def _exec(self, code: str) -> str:
        """Run code in the persistent namespace, capturing stdout."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<interpreter>", "exec"), self._namespace)
        except Exception:
            output = buf.getvalue()
            tb = traceback.format_exc()
            if output:
                return f"{output}\nERROR:\n{tb}"
            return f"ERROR:\n{tb}"

        output = buf.getvalue().strip()
        return output if output else "(no output)"


def _factory() -> PythonInterpreterTool:
    return PythonInterpreterTool()


register_tool("python_interpreter", _factory)
