"""Tests for agent.tools.skill_discover — skill discovery tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.skill_discover import SkillDiscoverTool


@pytest.fixture
def mock_manager():
    """Mock SkillManager."""
    manager = MagicMock()
    manager.discover_from_url = AsyncMock(return_value=[])
    manager.discover_from_search = AsyncMock(return_value=[])
    manager.bulk_review_and_integrate = AsyncMock(
        return_value={"discovered": 0, "approved": 0, "rejected": 0}
    )
    return manager


class TestSkillDiscoverTool:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        tool = SkillDiscoverTool()
        result = await tool.run("")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_url_discovery(self, mock_manager):
        tool = SkillDiscoverTool()
        tool._manager = mock_manager

        from agent.skills.models import Skill, SkillMetadata, SecurityReport

        skill = Skill(
            metadata=SkillMetadata(
                name="found-skill",
                description="A discovered skill",
                source_url="https://example.com",
                status="approved",
                fetched_at="2025-01-01T00:00:00Z",
                security_report=SecurityReport(
                    verdict="pass", score=9.0, findings=[], reviewed_at="2025-01-01T00:00:00Z"
                ),
            ),
            content="# Found Skill",
        )
        mock_manager.discover_from_url = AsyncMock(return_value=[skill])
        mock_manager.bulk_review_and_integrate = AsyncMock(
            return_value={"discovered": 1, "approved": 1, "rejected": 0}
        )

        result = await tool.run("https://github.com/anthropics/skills")
        assert "Discovered: 1" in result
        assert "Approved:   1" in result
        assert "found-skill" in result

    @pytest.mark.asyncio
    async def test_search_discovery(self, mock_manager):
        tool = SkillDiscoverTool()
        tool._manager = mock_manager

        result = await tool.run("PDF processing skills")
        assert "No skills found" in result or "Discovered: 0" in result

    @pytest.mark.asyncio
    async def test_no_manager(self):
        tool = SkillDiscoverTool()
        # Force _get_manager to return None
        tool._get_manager = lambda: None
        tool._manager = None

        result = await tool.run("https://example.com")
        assert "ERROR" in result
