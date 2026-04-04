"""Async client for Ollama /api/chat — zero third-party SDK, direct HTTP via httpx."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from agent.config import SamplingParams


# ---------------------------------------------------------------------------
# Thinking-mode helpers
# ---------------------------------------------------------------------------

_THINKING_PREFIX = "<|think|>"
_CHANNEL_PATTERN = re.compile(
    r"<\|channel>thought\s*\n.*?<channel\|>\s*", re.DOTALL
)


def _inject_thinking(system_prompt: str) -> str:
    """Prepend the thinking token if not already present."""
    if system_prompt.startswith(_THINKING_PREFIX):
        return system_prompt
    return f"{_THINKING_PREFIX}{system_prompt}"


def _strip_thinking(text: str) -> str:
    """Remove thinking channel blocks, returning only the final answer."""
    return _CHANNEL_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Cannot reach the Ollama server."""


class OllamaModelNotFound(OllamaError):
    """Requested model is not available."""


class OllamaTimeout(OllamaError):
    """Request timed out."""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChatResponse:
    content: str
    model: str = ""
    total_duration: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Async HTTP client for Ollama /api/chat."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # -- public API ----------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,
        thinking: bool = False,
        stream: bool = False,
    ) -> ChatResponse:
        """Send a chat completion request. Returns the full response."""
        messages = self._prepare_messages(messages, thinking)
        payload = self._build_payload(model, messages, sampling, stream=False)

        data = await self._post("/api/chat", payload)
        content = data.get("message", {}).get("content", "")
        if thinking:
            content = _strip_thinking(content)

        return ChatResponse(
            content=content,
            model=data.get("model", model),
            total_duration=data.get("total_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            raw=data,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,
        thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Stream chat tokens. Yields content chunks; strips thinking blocks at the end."""
        messages = self._prepare_messages(messages, thinking)
        payload = self._build_payload(model, messages, sampling, stream=True)

        buffer = ""
        async for chunk_data in self._post_stream("/api/chat", payload):
            token = chunk_data.get("message", {}).get("content", "")
            if token:
                buffer += token
                if not thinking:
                    yield token

        # For thinking mode, yield the cleaned final content at once
        if thinking:
            yield _strip_thinking(buffer)

    # -- internals -----------------------------------------------------------

    def _prepare_messages(
        self, messages: list[dict[str, str]], thinking: bool
    ) -> list[dict[str, str]]:
        """Inject thinking prefix into system prompt if needed."""
        if not thinking:
            return messages
        out: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                out.append({**msg, "content": _inject_thinking(msg["content"])})
            else:
                out.append(msg)
        return out

    def _build_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        sampling: SamplingParams | None,
        *,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if sampling:
            payload["options"] = {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
            }
        return payload

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}{path}", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Ollama request timed out") from exc

        if resp.status_code == 404:
            raise OllamaModelNotFound(f"Model not found: {payload.get('model')}")
        resp.raise_for_status()
        return resp.json()

    async def _post_stream(self, path: str, payload: dict) -> AsyncIterator[dict]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", f"{self._base_url}{path}", json=payload
                ) as resp:
                    if resp.status_code == 404:
                        raise OllamaModelNotFound(
                            f"Model not found: {payload.get('model')}"
                        )
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield json.loads(line)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Ollama request timed out") from exc
