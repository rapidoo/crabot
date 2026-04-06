"""Spider search tool — web scraping and search via spider.cloud."""

from __future__ import annotations

import logging
import os

import httpx

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


class SpiderSearchTool:
    """Search and scrape the web using Spider.cloud API."""

    API_URL = "https://api.spider.cloud"

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self._api_key = api_key or os.environ.get("SPIDER_API_KEY", "")
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "spider_search"

    @property
    def description(self) -> str:
        return (
            "Search the internet and scrape web pages for current information. "
            "Input can be a search query or a URL to scrape."
        )

    async def run(self, input: str) -> str:
        """Search the web or scrape a URL.

        Input: a search query string, or a URL to scrape.
        Returns: formatted search results or page content.
        """
        query = input.strip()
        if not query:
            return "ERROR: empty search query"

        if not self._api_key:
            return "ERROR: SPIDER_API_KEY not set in .env"

        # If input looks like a URL, scrape it; otherwise search
        if query.startswith("http://") or query.startswith("https://"):
            return await self._scrape(query)
        return await self._search(query)

    async def _search(self, query: str) -> str:
        """Search the web via Spider.cloud."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.API_URL}/search",
                    json={
                        "query": query,
                        "limit": self._max_results,
                        "return_format": "markdown",
                    },
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            if not data:
                return f"No results found for '{query}'"

            lines: list[str] = [f"Search results for '{query}':\n"]
            for i, r in enumerate(data, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                content = r.get("content", "")
                # Truncate content to keep response manageable
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"{i}. **{title}**")
                if url:
                    lines.append(f"   {url}")
                if content:
                    lines.append(f"   {content}")
                lines.append("")

            return "\n".join(lines).strip()

        except httpx.HTTPStatusError as exc:
            logger.error("Spider search HTTP error: %s", exc.response.status_code)
            return f"ERROR: Spider search returned {exc.response.status_code}"
        except Exception as exc:
            logger.error("Spider search failed: %s", exc)
            return f"ERROR: search failed — {exc}"

    async def _scrape(self, url: str) -> str:
        """Scrape a single URL via Spider.cloud."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.API_URL}/crawl",
                    json={
                        "url": url,
                        "limit": 1,
                        "return_format": "markdown",
                    },
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            if not data:
                return f"No content retrieved from {url}"

            page = data[0] if isinstance(data, list) else data
            title = page.get("title", url)
            content = page.get("content", "")

            # Truncate to avoid overwhelming the context
            if len(content) > 3000:
                content = content[:3000] + "\n\n... (truncated)"

            return f"**{title}**\n{url}\n\n{content}"

        except httpx.HTTPStatusError as exc:
            logger.error("Spider scrape HTTP error: %s", exc.response.status_code)
            return f"ERROR: Spider scrape returned {exc.response.status_code}"
        except Exception as exc:
            logger.error("Spider scrape failed: %s", exc)
            return f"ERROR: scrape failed — {exc}"


def _factory() -> SpiderSearchTool | None:
    api_key = os.environ.get("SPIDER_API_KEY", "")
    if not api_key:
        logger.warning("spider_search tool unavailable: SPIDER_API_KEY not set")
        return None
    return SpiderSearchTool(api_key=api_key)


register_tool("spider_search", _factory)
