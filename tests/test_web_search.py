"""Tests for web search tool (Brave Search API)."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from agent.tools.web_search import WebSearchTool


class TestWebSearchToolUnit:
    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool(api_key="fake")
        result = await tool.run("")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        tool = WebSearchTool(api_key="")
        result = await tool.run("test query")
        assert "BRAVE_API_KEY" in result

    def test_properties(self):
        tool = WebSearchTool(api_key="fake")
        assert tool.name == "web_search"
        assert "internet" in tool.description.lower()

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_search(self):
        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Python.org",
                                "url": "https://python.org",
                                "description": "Official Python website",
                            },
                            {
                                "title": "Python Wikipedia",
                                "url": "https://en.wikipedia.org/wiki/Python",
                                "description": "Python programming language",
                            },
                        ]
                    }
                },
            )
        )

        tool = WebSearchTool(api_key="fake_key", max_results=2)
        result = await tool.run("Python programming")
        assert "Python.org" in result
        assert "python.org" in result
        assert "Python Wikipedia" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_results(self):
        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )

        tool = WebSearchTool(api_key="fake_key")
        result = await tool.run("xyznonexistentquery123")
        assert "No results" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error(self):
        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(429, json={"error": "rate limited"})
        )

        tool = WebSearchTool(api_key="fake_key")
        result = await tool.run("test")
        assert "429" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_key_sent_in_header(self):
        route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )

        tool = WebSearchTool(api_key="my_secret_key")
        await tool.run("test")

        request = route.calls[0].request
        assert request.headers["X-Subscription-Token"] == "my_secret_key"


@pytest.mark.integration
class TestWebSearchToolIntegration:
    @pytest.mark.asyncio
    async def test_real_search(self):
        import os
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            pytest.skip("BRAVE_API_KEY not set")

        tool = WebSearchTool(api_key=api_key, max_results=3)
        result = await tool.run("Python Ollama local LLM")
        assert "results for" in result.lower()
        assert len(result) > 100
