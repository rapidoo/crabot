"""Planner — decomposes a user request into a structured JSON plan."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from agent.config import get_settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.infra.retry import with_retry, MaxRetriesExceeded
from agent.schemas import Plan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a task planner. Given a user request and memory context, produce
a JSON plan ONLY. No prose. No explanation.

Output format (strict):
{
  "goal": "one sentence description of the overall goal",
  "steps": [
    {
      "id": 1,
      "tool": "search|web_search|code|memory|file|none",
      "input": "exact input to pass to the tool or LLM",
      "expected_output": "what a correct result looks like"
    }
  ],
  "parallel": [1, 2]
}

Tools available:
- search: search local project files for keywords
- web_search: search the internet (DuckDuckGo) for current information
- code: execute Python code in a sandbox
- file: read or write local files (read:<path> or write:<path>:<content>)
- memory: query the memory graph
- none: use the LLM directly (no tool)

Rules:
- tool must be one of: search, web_search, code, memory, file, none
- Each step must have a unique id starting from 1
- parallel lists step ids that can run concurrently (optional, default empty)
- Output ONLY the JSON object, no markdown fences, no commentary"""

# Regex to extract JSON from markdown-fenced or prose-wrapped responses
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> str:
    """Try to extract a JSON object from potentially wrapped text."""
    text = text.strip()

    # Already clean JSON?
    if text.startswith("{"):
        return text

    # Wrapped in markdown fences?
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    # Embedded in prose?
    m = _JSON_OBJECT_RE.search(text)
    if m:
        return m.group(0)

    return text


def _parse_plan(raw: str) -> Plan:
    """Parse raw LLM output into a validated Plan."""
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return Plan.model_validate(data)


class Planner:
    """Generates a structured Plan from user input using the LLM."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        router: ModelRouter | None = None,
    ):
        settings = get_settings()
        self._client = client or OllamaClient(
            base_url=settings.models.ollama_base_url
        )
        self._router = router or ModelRouter(settings)

    async def plan(self, user_input: str, context: str = "") -> Plan:
        """Generate a plan for the given user input.

        Args:
            user_input: The user's request.
            context: Optional memory context to inject into the prompt.

        Returns:
            A validated Plan object.

        Raises:
            MaxRetriesExceeded: If the LLM fails to produce valid JSON after retries.
        """
        return await with_retry(
            self._attempt_plan,
            user_input,
            context,
            max_retries=3,
            base_ms=1000,
            retry_on=(json.JSONDecodeError, ValidationError),
        )

    async def _attempt_plan(self, user_input: str, context: str) -> Plan:
        """Single attempt at generating and parsing a plan."""
        messages = self._build_messages(user_input, context)

        model = self._router.select("planner")
        sampling = self._router.sampling("planner")
        thinking = self._router.thinking_enabled("planner")

        resp: ChatResponse = await self._client.chat(
            model, messages, sampling=sampling, thinking=thinking
        )

        logger.debug("Planner raw response: %s", resp.content[:500])
        return _parse_plan(resp.content)

    def _build_messages(
        self, user_input: str, context: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]
        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Memory context:\n{context}\n\nUser request:\n{user_input}",
                }
            )
        else:
            messages.append({"role": "user", "content": user_input})
        return messages
