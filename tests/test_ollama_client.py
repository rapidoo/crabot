"""Tests for OllamaClient — unit (mocked) and integration (real Ollama)."""

import json

import httpx
import pytest
import respx

from agent.config import SamplingParams
from agent.models.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFound,
    OllamaTimeout,
    ChatResponse,
    _inject_thinking,
    _strip_thinking,
)


# ---------------------------------------------------------------------------
# Thinking helpers
# ---------------------------------------------------------------------------


class TestThinkingHelpers:
    def test_inject_thinking(self):
        assert _inject_thinking("You are a planner.").startswith("<|think|>")

    def test_inject_thinking_idempotent(self):
        already = "<|think|>You are a planner."
        assert _inject_thinking(already) == already

    def test_strip_thinking_removes_block(self):
        text = (
            "<|channel>thought\nLet me think about this...\n<channel|>\n"
            "The answer is 42."
        )
        assert _strip_thinking(text) == "The answer is 42."

    def test_strip_thinking_no_block(self):
        text = "The answer is 42."
        assert _strip_thinking(text) == text

    def test_strip_thinking_multiple_blocks(self):
        text = (
            "<|channel>thought\nFirst thought\n<channel|>\n"
            "Part 1. "
            "<|channel>thought\nSecond thought\n<channel|>\n"
            "Part 2."
        )
        result = _strip_thinking(text)
        assert "Part 1." in result
        assert "Part 2." in result
        assert "thought" not in result.lower() or "thought" not in result


# ---------------------------------------------------------------------------
# Unit tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOllamaClientUnit:
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_basic(self):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Hello!"},
                    "model": "gemma4:e4b",
                    "total_duration": 1000,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )
        )

        client = OllamaClient()
        resp = await client.chat(
            "gemma4:e4b",
            [{"role": "user", "content": "Hi"}],
        )
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello!"
        assert resp.model == "gemma4:e4b"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_with_sampling(self):
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "ok"}},
            )
        )

        client = OllamaClient()
        await client.chat(
            "gemma4:26b",
            [{"role": "user", "content": "test"}],
            sampling=SamplingParams(temperature=0.7),
        )

        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["options"]["temperature"] == 0.7

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_thinking_injects_prefix(self):
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "content": (
                            "<|channel>thought\nreasoning\n<channel|>\nFinal answer"
                        )
                    }
                },
            )
        )

        client = OllamaClient()
        resp = await client.chat(
            "gemma4:26b",
            [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "Plan this"},
            ],
            thinking=True,
        )

        # Verify thinking prefix was injected
        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["messages"][0]["content"].startswith("<|think|>")

        # Verify thinking block was stripped
        assert resp.content == "Final answer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error(self):
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        client = OllamaClient()
        with pytest.raises(OllamaConnectionError):
            await client.chat("gemma4:e4b", [{"role": "user", "content": "hi"}])

    @respx.mock
    @pytest.mark.asyncio
    async def test_model_not_found(self):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )

        client = OllamaClient()
        with pytest.raises(OllamaModelNotFound):
            await client.chat("gemma4:fake", [{"role": "user", "content": "hi"}])

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout(self):
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )

        client = OllamaClient()
        with pytest.raises(OllamaTimeout):
            await client.chat("gemma4:e4b", [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Integration tests (require Ollama running + gemma4:e4b pulled)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOllamaClientIntegration:
    @pytest.mark.asyncio
    async def test_real_chat(self):
        client = OllamaClient()
        resp = await client.chat(
            "gemma4",
            [{"role": "user", "content": "Say hello in exactly one word."}],
            sampling=SamplingParams(temperature=0.3),
        )
        assert len(resp.content) > 0
        assert resp.eval_count > 0

    @pytest.mark.asyncio
    async def test_real_json_output(self):
        client = OllamaClient()
        resp = await client.chat(
            "gemma4",
            [
                {
                    "role": "system",
                    "content": "You output JSON only. No prose.",
                },
                {
                    "role": "user",
                    "content": 'Output: {"greeting": "hello", "number": 42}',
                },
            ],
            sampling=SamplingParams(temperature=0.1),
        )
        # Try to parse as JSON — this tests model reliability
        content = resp.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)  # Model produced valid JSON

    @pytest.mark.asyncio
    async def test_real_streaming(self):
        client = OllamaClient()
        chunks: list[str] = []
        async for chunk in client.chat_stream(
            "gemma4",
            [{"role": "user", "content": "Count from 1 to 5."}],
            sampling=SamplingParams(temperature=0.3),
        ):
            chunks.append(chunk)
        full = "".join(chunks)
        assert len(full) > 0
        assert len(chunks) > 1  # Should have multiple chunks
