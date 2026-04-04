"""Exponential backoff retry wrapper for Ollama calls."""

from __future__ import annotations

import asyncio
from typing import TypeVar, Callable, Awaitable, Any

from agent.models.ollama_client import OllamaError

T = TypeVar("T")

BASE_RETRY_MS = 2000
MAX_RETRIES = 4


class MaxRetriesExceeded(Exception):
    """All retry attempts exhausted."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed after {attempts} attempts: {last_error}")


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    base_ms: int = BASE_RETRY_MS,
    retry_on: type[Exception] | tuple[type[Exception], ...] = OllamaError,
) -> T:
    """Call *fn* with exponential backoff on retryable errors.

    Raises MaxRetriesExceeded if all attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn(*args)
        except retry_on as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = base_ms * (2 ** attempt) / 1000
                await asyncio.sleep(delay)

    assert last_error is not None
    raise MaxRetriesExceeded(max_retries, last_error)
