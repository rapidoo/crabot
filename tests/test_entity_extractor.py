"""Tests for EntityExtractor — unit and integration."""

import json
from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.memory.entity_extractor import EntityExtractor


class TestEntityExtractorUnit:
    def _make_extractor(self, response: str) -> EntityExtractor:
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content=response)
        )
        return EntityExtractor(client=client, router=ModelRouter(Settings()))

    @pytest.mark.asyncio
    async def test_valid_entities(self):
        entities_json = json.dumps([
            {"name": "Python", "type": "tech", "description": "programming language"},
            {"name": "Neo4j", "type": "tech", "description": "graph database"},
        ])
        extractor = self._make_extractor(entities_json)
        result = await extractor.extract("Tell me about Python and Neo4j")
        assert len(result) == 2
        assert result[0]["name"] == "Python"
        assert result[1]["name"] == "Neo4j"

    @pytest.mark.asyncio
    async def test_empty_array(self):
        extractor = self._make_extractor("[]")
        result = await extractor.extract("Hello world")
        assert result == []

    @pytest.mark.asyncio
    async def test_entities_in_prose(self):
        entities_json = json.dumps([
            {"name": "FastAPI", "type": "tech", "description": "web framework"},
        ])
        extractor = self._make_extractor(f"Here are the entities:\n{entities_json}")
        result = await extractor.extract("Build a FastAPI app")
        assert len(result) == 1
        assert result[0]["name"] == "FastAPI"

    @pytest.mark.asyncio
    async def test_garbage_response(self):
        extractor = self._make_extractor("I don't understand")
        result = await extractor.extract("Something")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_input(self):
        extractor = self._make_extractor("[]")
        result = await extractor.extract("")
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_fields_normalized(self):
        entities_json = json.dumps([
            {"name": "X"},  # missing type and description
        ])
        extractor = self._make_extractor(entities_json)
        result = await extractor.extract("About X")
        assert len(result) == 1
        assert result[0]["type"] == "other"
        assert result[0]["description"] == ""


@pytest.mark.integration
class TestEntityExtractorIntegration:
    @pytest.mark.asyncio
    async def test_real_extraction(self):
        settings = Settings()
        settings.models.triage = "gemma4"
        extractor = EntityExtractor(
            client=OllamaClient(base_url=settings.models.ollama_base_url),
            router=ModelRouter(settings),
        )
        result = await extractor.extract(
            "Implement a REST API using FastAPI and PostgreSQL for user authentication"
        )
        assert len(result) >= 1
        names = [e["name"].lower() for e in result]
        # Should find at least one of the key entities
        found = any(
            keyword in name
            for name in names
            for keyword in ("fastapi", "postgresql", "api", "rest", "auth")
        )
        assert found, f"Expected tech entities, got: {names}"
