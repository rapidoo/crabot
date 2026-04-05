"""Triage — fast classification of input complexity using E4B."""

from __future__ import annotations

import logging
from typing import Literal

from agent.config import get_settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter

logger = logging.getLogger(__name__)

Complexity = Literal["simple", "complex", "multimodal"]

TRIAGE_SYSTEM_PROMPT = """\
You are a task complexity classifier. Given a user input, classify it into
exactly one category. Output ONLY the category name, nothing else.

Categories:
- simple: factual question, greeting, single-step task, short answer expected
- complex: multi-step task, code generation, analysis, planning, research
- multimodal: involves images, audio, or document processing

Output ONE word only: simple, complex, or multimodal."""


class Triage:
    """Classify input complexity to decide the pipeline path."""

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

    async def classify(self, user_input: str) -> Complexity:
        """Classify user input complexity.

        Uses E4B with thinking off for fast classification (< 1s target).
        """
        model = self._router.select("triage")
        sampling = self._router.sampling("triage")

        resp = await self._client.chat(
            model,
            [
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            sampling=sampling,
            thinking=self._router.thinking_enabled("triage"),
        )

        raw = resp.content.strip().lower()
        # Extract the classification from potentially verbose output
        for category in ("simple", "complex", "multimodal"):
            if category in raw:
                logger.info("Triage: '%s' → %s", user_input[:60], category)
                return category  # type: ignore[return-value]

        # Default to complex if unclear
        logger.warning("Triage unclear ('%s'), defaulting to complex", raw[:50])
        return "complex"
