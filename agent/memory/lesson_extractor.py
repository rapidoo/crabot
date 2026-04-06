"""Lesson extractor — detects user corrections and extracts actionable lessons."""

from __future__ import annotations

import json
import logging
import re

from agent.config import get_settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter

logger = logging.getLogger(__name__)

DETECT_SYSTEM_PROMPT = """\
You analyze conversations to detect when a user is correcting or teaching the AI.

Correction signals: disagreement ("non", "pas comme ça", "c'est faux"), \
instruction ("rappelle-toi", "la prochaine fois", "toujours", "jamais", "il faut"), \
preference ("je préfère", "utilise plutôt", "ne fais plus").

If a correction or lesson is present, return:
{"detected": true, "rule": "imperative action rule", "context": "when this applies", \
"category": "preference|correction|fact|process", "source_quote": "relevant user quote"}

If no correction or lesson is detected, return:
{"detected": false}

JSON ONLY. No prose."""

FEEDBACK_SYSTEM_PROMPT = """\
The user gave negative feedback on an AI response. Extract the lesson.

Return: {"rule": "imperative action rule", "context": "when this applies", \
"category": "preference|correction|fact|process", "source_quote": "user feedback verbatim"}

JSON ONLY. No prose."""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class LessonExtractor:
    """Detect user corrections and extract structured lessons using E4B."""

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

    async def detect_and_extract(
        self,
        user_message: str,
        assistant_response: str,
    ) -> dict | None:
        """Detect if user is correcting the assistant, extract lesson if so.

        Returns {"rule", "context", "category", "source_quote"} or None.
        """
        if not user_message.strip():
            return None

        model = self._router.select("triage")
        sampling = self._router.sampling("triage")

        prompt = (
            f"Assistant said: {assistant_response[:1000]}\n\n"
            f"User replied: {user_message[:1000]}"
        )

        try:
            resp = await self._client.chat(
                model,
                [
                    {"role": "system", "content": DETECT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                sampling=sampling,
                thinking=False,
            )
            return self._parse_detection(resp.content)
        except Exception as exc:
            logger.warning("Lesson detection failed: %s", exc)
            return None

    async def extract_from_feedback(
        self,
        reason: str,
        episode_summary: str,
    ) -> dict | None:
        """Extract a lesson from explicit /bad feedback reason.

        Returns {"rule", "context", "category", "source_quote"} or None.
        """
        if not reason.strip():
            return None

        model = self._router.select("triage")
        sampling = self._router.sampling("triage")

        prompt = (
            f"Episode summary: {episode_summary[:500]}\n"
            f"User feedback: {reason[:500]}"
        )

        try:
            resp = await self._client.chat(
                model,
                [
                    {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                sampling=sampling,
                thinking=False,
            )
            return self._parse_lesson(resp.content)
        except Exception as exc:
            logger.warning("Lesson extraction from feedback failed: %s", exc)
            return None

    def _parse_detection(self, raw: str) -> dict | None:
        """Parse detection response — returns lesson dict or None."""
        data = self._parse_json(raw)
        if data is None:
            return None
        if not data.get("detected", False):
            return None
        return self._validate_lesson(data)

    def _parse_lesson(self, raw: str) -> dict | None:
        """Parse a lesson extraction response."""
        data = self._parse_json(raw)
        if data is None:
            return None
        return self._validate_lesson(data)

    def _parse_json(self, raw: str) -> dict | None:
        """Extract JSON object from LLM response."""
        text = raw.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.debug("Could not parse lesson JSON from: %s", text[:200])
        return None

    def _validate_lesson(self, data: dict) -> dict | None:
        """Validate and normalize a lesson dict."""
        rule = str(data.get("rule", "")).strip()
        if not rule:
            return None
        return {
            "rule": rule,
            "context": str(data.get("context", "")).strip(),
            "category": str(data.get("category", "correction")).strip(),
            "source_quote": str(data.get("source_quote", "")).strip(),
        }
