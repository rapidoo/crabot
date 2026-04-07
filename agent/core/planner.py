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
from agent.intelligence.prompt_manager import PromptManager
from agent.personality.loader import Personality
from agent.schemas import Plan
from agent.tools.registry import list_tools_with_descriptions

logger = logging.getLogger(__name__)

# Default template used when no evolved prompt is available
_PLANNER_SYSTEM_TEMPLATE = """\
You are a task planner. Given a user request and memory context, produce
a JSON plan ONLY. No prose. No explanation.

Output format (strict):
{{
  "goal": "one sentence description of the overall goal",
  "steps": [
    {{
      "id": 1,
      "tool": "<tool_name>",
      "input": "exact input to pass to the tool or LLM",
      "expected_output": "what a correct result looks like"
    }}
  ],
  "parallel": [1, 2]
}}

Tools available:
{tools_section}
- none: use the LLM directly (no tool)

Tool routing hints (IMPORTANT — use the right tool for the job):
- User asks to CREATE, ADD, or BUILD a new tool/capability → use tool_create
- User asks to SEARCH the web or find current info → use web_search
- User asks to RUN or EXECUTE code → use code or python_interpreter
- User asks to READ, WRITE, or MODIFY files → use file
- User asks to SEARCH local files → use search
- Only use "none" when no tool fits (e.g. answering a question from memory, summarizing)

Rules:
- tool must be one of: {tool_names}, none
- Each step must have a unique id starting from 1
- parallel lists step ids that can run concurrently (optional, default empty)
- Output ONLY the JSON object, no markdown fences, no commentary"""

# Regex to extract JSON from markdown-fenced or prose-wrapped responses
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_system_prompt() -> str:
    """Build the planner system prompt with dynamically discovered tools."""
    tools = list_tools_with_descriptions()
    if tools:
        tools_section = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )
        tool_names = ", ".join(t["name"] for t in tools)
    else:
        tools_section = "- (no tools registered)"
        tool_names = ""
    return _PLANNER_SYSTEM_TEMPLATE.format(
        tools_section=tools_section,
        tool_names=tool_names,
    )


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
        personality: Personality | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        settings = get_settings()
        self._client = client or OllamaClient(
            base_url=settings.models.ollama_base_url
        )
        self._router = router or ModelRouter(settings)
        self._personality = personality
        self._prompt_manager = prompt_manager

    async def plan(self, user_input: str, context: str = "") -> Plan:
        """Generate a plan for the given user input."""
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
        # Use evolved prompt if available
        system_prompt = await self._get_system_prompt()
        if self._personality and self._personality.agents:
            system_prompt += f"\n\nOperational rules:\n{self._personality.agents}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
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

        model = self._router.select("planner")
        sampling = self._router.sampling("planner")
        thinking = self._router.thinking_enabled("planner")

        resp: ChatResponse = await self._client.chat(
            model, messages, sampling=sampling, thinking=thinking
        )

        logger.debug("Planner raw response: %s", resp.content[:500])
        return _parse_plan(resp.content)

    async def _get_system_prompt(self) -> str:
        """Get the system prompt, checking PromptManager for an evolved version."""
        default = _build_system_prompt()
        if self._prompt_manager and hasattr(self._prompt_manager, "get_prompt"):
            try:
                evolved = await self._prompt_manager.get_prompt("planner", default)
                if evolved != default:
                    logger.debug("Using evolved planner prompt")
                return evolved
            except Exception as exc:
                logger.warning("Failed to get evolved planner prompt: %s", exc)
        return default

