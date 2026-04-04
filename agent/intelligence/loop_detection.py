"""Loop detection — prevents infinite tool call loops.

3 detectors inspired by OpenClaw:
- Repeat: same tool + same input N times consecutively
- Circuit Breaker: total tool calls exceed threshold
- Ping-Pong: alternating between two tools repeatedly
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)

MAX_REPEAT = 3
CIRCUIT_BREAKER = 20
PING_PONG_DEPTH = 4


class LoopDetector:
    """Detects and prevents infinite tool call loops during execution."""

    def __init__(
        self,
        max_repeat: int = MAX_REPEAT,
        circuit_breaker: int = CIRCUIT_BREAKER,
        ping_pong_depth: int = PING_PONG_DEPTH,
    ):
        self._max_repeat = max_repeat
        self._circuit_breaker = circuit_breaker
        self._ping_pong_depth = ping_pong_depth
        self._history: list[tuple[str, str]] = []  # (tool_name, input_hash)
        self._total_calls = 0

    def check(self, tool_name: str, tool_input: str) -> str | None:
        """Check if this tool call would create a loop.

        Returns None if safe, or a string describing the detected loop.
        """
        input_hash = str(hash(tool_input))[:12]
        entry = (tool_name, input_hash)

        self._total_calls += 1
        self._history.append(entry)

        # Detector 1: Repeat — same tool + same input N times
        if len(self._history) >= self._max_repeat:
            recent = self._history[-self._max_repeat:]
            if all(r == entry for r in recent):
                msg = f"Repeat loop: {tool_name} called {self._max_repeat}x with same input"
                logger.warning("Loop detected: %s", msg)
                return msg

        # Detector 2: Circuit Breaker — too many total calls
        if self._total_calls > self._circuit_breaker:
            msg = f"Circuit breaker: {self._total_calls} tool calls exceeded limit {self._circuit_breaker}"
            logger.warning("Loop detected: %s", msg)
            return msg

        # Detector 3: Ping-Pong — alternating A→B→A→B
        if len(self._history) >= self._ping_pong_depth * 2:
            recent = self._history[-(self._ping_pong_depth * 2):]
            tools = [r[0] for r in recent]
            unique = set(tools)
            if len(unique) == 2:
                a, b = list(unique)
                expected = [a, b] * self._ping_pong_depth
                alt1 = [b, a] * self._ping_pong_depth
                if tools == expected or tools == alt1:
                    msg = f"Ping-pong loop: alternating {a} ↔ {b}"
                    logger.warning("Loop detected: %s", msg)
                    return msg

        return None

    def reset(self) -> None:
        """Reset state for a new run."""
        self._history.clear()
        self._total_calls = 0
