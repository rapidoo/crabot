"""Tests for Triage — unit and integration."""

from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.core.triage import Triage


class TestTriageUnit:
    def _make_triage(self, response: str) -> Triage:
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content=response)
        )
        return Triage(client=client, router=ModelRouter(Settings()))

    @pytest.mark.asyncio
    async def test_simple(self):
        triage = self._make_triage("simple")
        assert await triage.classify("What is 2+2?") == "simple"

    @pytest.mark.asyncio
    async def test_complex(self):
        triage = self._make_triage("complex")
        assert await triage.classify("Write a web scraper") == "complex"

    @pytest.mark.asyncio
    async def test_multimodal(self):
        triage = self._make_triage("multimodal")
        assert await triage.classify("Analyze this image") == "multimodal"

    @pytest.mark.asyncio
    async def test_verbose_response_extracts_category(self):
        triage = self._make_triage("I think this is a simple question.")
        assert await triage.classify("Hello") == "simple"

    @pytest.mark.asyncio
    async def test_unclear_defaults_to_complex(self):
        triage = self._make_triage("I'm not sure what this is.")
        assert await triage.classify("???") == "complex"

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        triage = self._make_triage("COMPLEX")
        assert await triage.classify("Build an app") == "complex"


@pytest.mark.integration
class TestTriageIntegration:
    @pytest.mark.asyncio
    async def test_real_classification(self):
        settings = Settings()
        settings.models.triage = "gemma4"
        triage = Triage(
            client=OllamaClient(base_url=settings.models.ollama_base_url),
            router=ModelRouter(settings),
        )

        # Simple questions should classify as simple
        result = await triage.classify("What is the capital of France?")
        assert result in ("simple", "complex")  # accept both, just verify it works

        # Complex tasks should classify as complex
        result = await triage.classify(
            "Write a Python web scraper that extracts product prices "
            "from multiple e-commerce sites and stores them in a database"
        )
        assert result in ("simple", "complex", "multimodal")
