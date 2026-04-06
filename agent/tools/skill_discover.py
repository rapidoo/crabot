"""Skill discovery tool — lets the agent find and integrate skills from the web."""

from __future__ import annotations

import logging

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


class SkillDiscoverTool:
    """Discover, review, and integrate new skills from the internet.

    Input: a URL to a skill repository, a direct SKILL.md URL, or a search query.
    The tool will:
    1. Fetch skill definitions via spider.cloud
    2. Run a security review on each
    3. Integrate approved skills into the local library
    """

    def __init__(self) -> None:
        self._manager = None  # Lazily initialized

    @property
    def name(self) -> str:
        return "skill_discover"

    @property
    def description(self) -> str:
        return (
            "Discover and integrate AI skills from the internet. "
            "Input: a URL (GitHub repo or SKILL.md) or a search query. "
            "Automatically runs security review before integration."
        )

    def _get_manager(self):
        """Lazily initialize the SkillManager with required dependencies."""
        if self._manager is not None:
            return self._manager

        from agent.skills.manager import SkillManager
        from agent.skills.store import SkillStore
        from agent.tools.registry import get_tool

        store = SkillStore()
        security_tool = get_tool("security_review")
        spider_tool = get_tool("spider_search")

        if security_tool is None:
            return None

        self._manager = SkillManager(
            store=store,
            security_tool=security_tool,
            spider_tool=spider_tool,
        )
        return self._manager

    async def run(self, input: str) -> str:
        """Execute skill discovery and integration."""
        query = input.strip()
        if not query:
            return "ERROR: provide a URL or search query"

        manager = self._get_manager()
        if manager is None:
            return "ERROR: required tools (security_review) not available"

        # Determine if input is a URL or search query
        if query.startswith("http://") or query.startswith("https://"):
            skills = await manager.discover_from_url(query)
            source = f"URL: {query}"
        else:
            skills = await manager.discover_from_search(query)
            source = f"search: {query}"

        if not skills:
            return f"No skills found from {source}"

        # Review and integrate
        stats = await manager.bulk_review_and_integrate(skills)

        # Build summary
        lines = [
            f"Skill discovery from {source}:",
            f"  Discovered: {stats['discovered']}",
            f"  Approved:   {stats['approved']}",
            f"  Rejected:   {stats['rejected']}",
            "",
        ]

        for skill in skills:
            status = skill.metadata.status
            name = skill.metadata.name
            score = (
                skill.metadata.security_report.score
                if skill.metadata.security_report
                else 0
            )
            lines.append(f"  [{status.upper()}] {name} (security score: {score:.1f})")

        return "\n".join(lines)


def _factory() -> SkillDiscoverTool | None:
    # Always available — spider_search checked lazily
    return SkillDiscoverTool()


register_tool("skill_discover", _factory)
