"""Code execution tool — sandboxed Python via subprocess."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from agent.tools.registry import register_tool

DEFAULT_TIMEOUT = 30


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class CodeExecTool:
    """Execute Python code in a sandboxed subprocess."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Execute Python code in an isolated subprocess."

    async def run(self, input: str) -> str:
        """Execute the given Python code and return stdout/stderr."""
        result = await self._sandbox_exec(input)
        if result.returncode != 0:
            output = f"EXIT CODE: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}"
            return output.strip()
        return result.stdout.strip()

    async def _sandbox_exec(self, code: str) -> ExecResult:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        # Remove potentially dangerous env vars
        for key in ("PYTHONSTARTUP", "PYTHONPATH"):
            env.pop(key, None)

        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
            return ExecResult(
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(returncode=-1, stdout="", stderr="TIMEOUT")


def _factory() -> CodeExecTool:
    return CodeExecTool()


register_tool("code", _factory)
