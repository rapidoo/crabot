"""Tests for exponential backoff retry wrapper."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent.models.ollama_client import OllamaError, OllamaConnectionError
from agent.infra.retry import with_retry, MaxRetriesExceeded


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        fn = AsyncMock(return_value="ok")
        result = await with_retry(fn, max_retries=3, base_ms=10)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self):
        fn = AsyncMock(
            side_effect=[OllamaConnectionError("fail"), OllamaError("fail"), "ok"]
        )
        result = await with_retry(fn, max_retries=3, base_ms=10)
        assert result == "ok"
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        fn = AsyncMock(side_effect=OllamaError("always fails"))
        with pytest.raises(MaxRetriesExceeded) as exc_info:
            await with_retry(fn, max_retries=3, base_ms=10)
        assert exc_info.value.attempts == 3
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        fn = AsyncMock(side_effect=ValueError("not retryable"))
        with pytest.raises(ValueError, match="not retryable"):
            await with_retry(fn, max_retries=3, base_ms=10)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retry_on(self):
        fn = AsyncMock(side_effect=[ValueError("retry me"), "ok"])
        result = await with_retry(
            fn, max_retries=3, base_ms=10, retry_on=ValueError
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_backoff_timing(self):
        """Verify that delays increase exponentially (roughly)."""
        fn = AsyncMock(side_effect=[OllamaError("1"), OllamaError("2"), "ok"])

        loop = asyncio.get_event_loop()
        start = loop.time()
        await with_retry(fn, max_retries=3, base_ms=50)
        elapsed = loop.time() - start

        # base_ms=50: delay 0 = 50ms, delay 1 = 100ms → total ~150ms minimum
        assert elapsed >= 0.1  # at least 100ms (accounting for variance)

    @pytest.mark.asyncio
    async def test_passes_args(self):
        fn = AsyncMock(return_value="ok")
        await with_retry(fn, "a", "b", max_retries=2, base_ms=10)
        fn.assert_called_with("a", "b")
