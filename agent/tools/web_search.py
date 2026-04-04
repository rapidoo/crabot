"""Web search tool — internet search via Brave Search API."""

from __future__ import annotations

import logging
import os

import httpx

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Search the internet using Brave Search API."""

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the internet for current information."

    async def run(self, input: str) -> str:
        """Search the web for the given query.

        Input: a search query string.
        Returns: formatted search results with titles, URLs, and snippets.
        """
        query = input.strip()
        if not query:
            return "ERROR: empty search query"

        if not self._api_key:
            return "ERROR: BRAVE_API_KEY not set in .env"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    self.API_URL,
                    params={"q": query, "count": self._max_results},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                return f"No results found for '{query}'"

            lines: list[str] = [f"Search results for '{query}':\n"]
            for i, r in enumerate(web_results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"{i}. **{title}**")
                if url:
                    lines.append(f"   {url}")
                if desc:
                    lines.append(f"   {desc}")
                lines.append("")

            return "\n".join(lines).strip()

        except httpx.HTTPStatusError as exc:
            logger.error("Brave Search HTTP error: %s", exc.response.status_code)
            return f"ERROR: Brave Search returned {exc.response.status_code}"
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return f"ERROR: search failed — {exc}"


def _factory() -> WebSearchTool | None:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        logger.warning("web_search tool unavailable: BRAVE_API_KEY not set")
        return None
    return WebSearchTool(api_key=api_key)


register_tool("web_search", _factory)
