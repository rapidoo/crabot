"""Output truncation — prevents oversized tool outputs from blowing context budget."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_CHARS = 50_000
DEFAULT_TAIL_CHARS = 2_000


def truncate_output(
    output: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> str:
    """Truncate tool output if it exceeds max_chars.

    Keeps the beginning and end of the output, replacing the middle
    with a summary indicator showing how much was cut.

    Args:
        output: Raw tool output string.
        max_chars: Maximum allowed characters.
        tail_chars: Number of characters to preserve from the end.

    Returns:
        Original output if within budget, or truncated version.
    """
    if len(output) <= max_chars:
        return output

    cut = len(output) - max_chars
    head_chars = max_chars - tail_chars

    head = output[:head_chars]
    tail = output[-tail_chars:]

    logger.info(
        "Output truncated: %d chars → %d chars (cut %d from middle)",
        len(output),
        max_chars,
        cut,
    )

    return (
        f"{head}\n\n"
        f"[... {cut:,} characters truncated ...]\n\n"
        f"{tail}"
    )
