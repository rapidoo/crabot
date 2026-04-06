"""Approval gate — ask user confirmation before dangerous tool actions.

Works in interactive mode (REPL). In daemon/non-interactive mode,
falls back to auto-approve with logging.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Tools that always require approval
_DANGEROUS_TOOLS: set[str] = {"code", "tool_create"}

# Patterns in file tool input that require approval
_DANGEROUS_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^write:", re.IGNORECASE),
    re.compile(r"^delete:", re.IGNORECASE),
]


class ApprovalGate:
    """Gate that checks tool calls and prompts for user approval if dangerous.

    In non-interactive mode (daemon, one-shot), dangerous actions are
    auto-approved but logged as warnings.
    """

    def __init__(self, *, interactive: bool = False, enabled: bool = True):
        self._interactive = interactive
        self._enabled = enabled
        self._approved_patterns: set[str] = set()

    def needs_approval(self, tool_name: str, tool_input: str) -> bool:
        """Check if a tool call requires user approval."""
        if not self._enabled:
            return False

        # Check cache of previously approved patterns
        cache_key = f"{tool_name}:{self._pattern_key(tool_input)}"
        if cache_key in self._approved_patterns:
            return False

        if tool_name in _DANGEROUS_TOOLS:
            return True

        if tool_name == "file":
            for pattern in _DANGEROUS_FILE_PATTERNS:
                if pattern.search(tool_input):
                    return True

        return False

    def request_approval(self, tool_name: str, tool_input: str) -> bool:
        """Request user approval for a tool call.

        Returns True if approved, False if denied.
        In non-interactive mode, auto-approves with a warning log.
        """
        if not self._interactive:
            logger.warning(
                "Auto-approving dangerous action (non-interactive): %s(%s)",
                tool_name,
                tool_input[:100],
            )
            return True

        # Show the action and ask for confirmation
        preview = tool_input[:200]
        if len(tool_input) > 200:
            preview += "..."

        print(f"\n  ⚠ Action requiring approval:")
        print(f"    Tool:  {tool_name}")
        print(f"    Input: {preview}")

        try:
            answer = input("    Approve? [y/N/always] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if answer in ("a", "always"):
            cache_key = f"{tool_name}:{self._pattern_key(tool_input)}"
            self._approved_patterns.add(cache_key)
            logger.info("Tool '%s' auto-approved for future calls", tool_name)
            return True

        return answer in ("y", "yes", "o", "oui")

    @staticmethod
    def _pattern_key(tool_input: str) -> str:
        """Extract a generalizable pattern from tool input for caching.

        For file ops, use the path. For code, use 'code'.
        """
        # For file operations, extract the path
        if ":" in tool_input:
            parts = tool_input.split(":", 2)
            if len(parts) >= 2:
                return parts[0] + ":" + parts[1]
        return tool_input[:50]
