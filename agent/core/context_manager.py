"""Context manager — multi-turn conversation history + context compression.

Maintains a sliding window of conversation history and compresses
old exchanges when approaching the token budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters (conservative for multilingual)
_CHARS_PER_TOKEN = 4


@dataclass
class Turn:
    """A single user→assistant exchange."""
    user: str
    assistant: str

    def char_count(self) -> int:
        return len(self.user) + len(self.assistant)


@dataclass
class ContextManager:
    """Maintains conversation history with automatic compression.

    When the estimated token count exceeds ``compression_threshold``
    (fraction of ``max_tokens``), older turns are summarized into a
    compact block to free up context budget.
    """

    max_tokens: int = 32768
    compression_threshold: float = 0.50
    protect_last_n: int = 4
    summary_block: str = ""
    turns: list[Turn] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ContextManager":
        s = settings or get_settings()
        return cls(
            max_tokens=s.context.max_tokens_per_call,
            compression_threshold=s.context.compression_threshold,
            protect_last_n=s.context.compression_protect_last_n,
        )

    # -- Public API -----------------------------------------------------------

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a completed exchange and compress if needed."""
        self.turns.append(Turn(user=user, assistant=assistant))
        if self._needs_compression():
            self._compress()

    def get_history_messages(self) -> list[dict[str, str]]:
        """Return conversation history as a list of message dicts.

        Includes a summary of older turns (if any) followed by the
        recent turns kept in full.
        """
        messages: list[dict[str, str]] = []
        if self.summary_block:
            messages.append({
                "role": "system",
                "content": (
                    "Summary of earlier conversation:\n"
                    f"{self.summary_block}"
                ),
            })
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def estimated_tokens(self) -> int:
        """Rough token estimate for the current history."""
        total_chars = len(self.summary_block)
        total_chars += sum(t.char_count() for t in self.turns)
        return total_chars // _CHARS_PER_TOKEN

    def clear(self) -> None:
        """Reset conversation history."""
        self.turns.clear()
        self.summary_block = ""

    # -- Internal -------------------------------------------------------------

    def _needs_compression(self) -> bool:
        budget = int(self.max_tokens * self.compression_threshold)
        return self.estimated_tokens() > budget and len(self.turns) > self.protect_last_n

    def _compress(self) -> None:
        """Compress older turns into a summary block, keeping the last N."""
        if len(self.turns) <= self.protect_last_n:
            return

        # Split: old turns to summarize, recent turns to keep
        cut = len(self.turns) - self.protect_last_n
        old_turns = self.turns[:cut]
        self.turns = self.turns[cut:]

        # Build summary from old turns
        summaries: list[str] = []
        if self.summary_block:
            summaries.append(self.summary_block)
        for t in old_turns:
            user_short = t.user[:200].replace("\n", " ")
            assistant_short = t.assistant[:300].replace("\n", " ")
            summaries.append(f"- User: {user_short} → Assistant: {assistant_short}")

        self.summary_block = "\n".join(summaries)
        logger.info(
            "Context compressed: %d old turns summarized, %d kept",
            len(old_turns),
            len(self.turns),
        )
