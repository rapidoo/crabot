"""Search tool — local keyword-based file search."""

from __future__ import annotations

import os
from pathlib import Path

from agent.tools.registry import register_tool


class SearchTool:
    """Search for files and content matching a keyword query."""

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root).resolve() if root else Path.cwd().resolve()

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search local files for matching content."

    async def run(self, input: str) -> str:
        """Search for files containing the query string.

        Input: a keyword or phrase to search for.
        Returns: matching file paths and relevant lines.
        """
        query = input.strip().lower()
        if not query:
            return "ERROR: empty search query"

        matches: list[str] = []
        max_results = 20
        max_file_size = 1_000_000  # 1MB

        for dirpath, _, filenames in os.walk(self._root):
            # Skip hidden dirs and common non-text dirs
            if any(part.startswith(".") for part in Path(dirpath).parts):
                continue
            if any(skip in dirpath for skip in ("node_modules", "__pycache__", ".venv")):
                continue

            for filename in filenames:
                if len(matches) >= max_results:
                    break

                filepath = Path(dirpath) / filename
                # Only search text-like files
                if filepath.suffix not in (
                    ".py", ".md", ".txt", ".yaml", ".yml", ".json",
                    ".toml", ".cfg", ".ini", ".sh", ".html", ".css", ".js",
                    ".ts", ".cypher", ".sql", ".csv",
                ):
                    continue

                try:
                    if filepath.stat().st_size > max_file_size:
                        continue
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError):
                    continue

                if query in content.lower():
                    # Find matching lines
                    lines = content.splitlines()
                    matching_lines = [
                        f"  L{i+1}: {line.strip()}"
                        for i, line in enumerate(lines)
                        if query in line.lower()
                    ][:5]  # max 5 lines per file

                    rel = filepath.relative_to(self._root)
                    matches.append(f"{rel}\n" + "\n".join(matching_lines))

        if not matches:
            return f"No results found for '{input}'"

        return f"Found {len(matches)} files matching '{input}':\n\n" + "\n\n".join(matches)


def _factory() -> SearchTool:
    return SearchTool()


register_tool("search", _factory)
