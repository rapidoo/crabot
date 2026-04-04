"""Entity extractor — uses E4B to extract named entities from text."""

from __future__ import annotations

import json
import logging
import re

from agent.config import get_settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """\
Extract named entities from the text. Return a JSON array ONLY. No prose.

Each entity: {"name": "...", "type": "person|org|tech|concept|place|other", "description": "one sentence"}

If no entities found, return: []"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class EntityExtractor:
    """Extract entities from text using a fast LLM call."""

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

    async def extract(self, text: str) -> list[dict[str, str]]:
        """Extract entities from the given text.

        Returns a list of {"name", "type", "description"} dicts.
        Returns empty list on failure.
        """
        if not text.strip():
            return []

        model = self._router.select("triage")  # E4B, fast
        sampling = self._router.sampling("triage")

        try:
            resp = await self._client.chat(
                model,
                [
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                sampling=sampling,
                thinking=False,
            )
            return self._parse_entities(resp.content)
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return []

    def _parse_entities(self, raw: str) -> list[dict[str, str]]:
        """Parse the LLM response into a list of entity dicts."""
        text = raw.strip()

        # Try direct parse
        if text.startswith("["):
            try:
                return self._validate(json.loads(text))
            except json.JSONDecodeError:
                pass

        # Extract array from prose
        m = _JSON_ARRAY_RE.search(text)
        if m:
            try:
                return self._validate(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass

        logger.debug("Could not parse entities from: %s", text[:200])
        return []

    def _validate(self, data: list) -> list[dict[str, str]]:
        """Validate and normalize entity list."""
        entities: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                entities.append({
                    "name": str(item.get("name", "")),
                    "type": str(item.get("type", "other")),
                    "description": str(item.get("description", "")),
                })
        return entities
