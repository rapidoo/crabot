"""File I/O tool — read and write files with path restriction."""

from __future__ import annotations

import os
from pathlib import Path

from agent.tools.registry import register_tool


class FileIOTool:
    """Read or write files within an allowed directory."""

    def __init__(self, allowed_root: str | Path | None = None):
        self._root = Path(allowed_root).resolve() if allowed_root else Path.cwd().resolve()

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Read or write files on the local filesystem."

    async def run(self, input: str) -> str:
        """Execute a file operation.

        Input format:
            read:<path>
            write:<path>:<content>
        """
        if input.startswith("read:"):
            return await self._read(input[5:].strip())
        elif input.startswith("write:"):
            parts = input[6:].split(":", 1)
            if len(parts) != 2:
                return "ERROR: write format is write:<path>:<content>"
            return await self._write(parts[0].strip(), parts[1])
        else:
            return f"ERROR: unknown file operation. Use read:<path> or write:<path>:<content>"

    async def _read(self, path_str: str) -> str:
        path = self._resolve_safe(path_str)
        if path is None:
            return f"ERROR: path '{path_str}' is outside allowed root '{self._root}'"
        if not path.exists():
            return f"ERROR: file not found: {path}"
        return path.read_text(encoding="utf-8", errors="replace")

    async def _write(self, path_str: str, content: str) -> str:
        path = self._resolve_safe(path_str)
        if path is None:
            return f"ERROR: path '{path_str}' is outside allowed root '{self._root}'"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"

    def _resolve_safe(self, path_str: str) -> Path | None:
        """Resolve path and ensure it stays within the allowed root."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self._root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            return None
        return resolved


def _factory() -> FileIOTool:
    # Use repo root as allowed_root so the agent can access its own codebase
    repo_root = Path(__file__).resolve().parent.parent.parent
    return FileIOTool(allowed_root=repo_root)


register_tool("file", _factory)
