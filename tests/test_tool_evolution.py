"""Tests for tool evolution — lifecycle management of generated tools."""

from unittest.mock import AsyncMock

import pytest

from agent.intelligence.tool_evolution import (
    track_tool_usage,
    evaluate_tools,
    MIN_USES_FOR_DISABLE,
    DISABLE_SCORE_THRESHOLD,
    PROMOTE_SCORE_THRESHOLD,
    MAX_ERROR_RATE,
)


@pytest.fixture
def memory():
    m = AsyncMock()
    m.available = True
    return m


class TestTrackUsage:
    @pytest.mark.asyncio
    async def test_tracks_success(self, memory):
        await track_tool_usage(memory, "test_tool", success=True, score=8.0)
        memory.update_tool_stats.assert_awaited_once_with(
            "test_tool", success=True, score=8.0
        )

    @pytest.mark.asyncio
    async def test_unavailable_memory_noop(self):
        m = AsyncMock()
        m.available = False
        await track_tool_usage(m, "tool", success=True)
        m.update_tool_stats.assert_not_awaited()


class TestEvaluateTools:
    @pytest.mark.asyncio
    async def test_disable_high_error_rate(self, memory):
        memory.get_generated_tools = AsyncMock(return_value=[
            {"name": "bad_tool", "usage_count": 10, "error_count": 5, "avg_score": 6.0},
        ])
        actions = await evaluate_tools(memory)
        assert len(actions) == 1
        assert actions[0]["action"] == "disable"
        assert "bad_tool" in actions[0]["tool"]

    @pytest.mark.asyncio
    async def test_disable_low_score(self, memory):
        memory.get_generated_tools = AsyncMock(return_value=[
            {"name": "poor_tool", "usage_count": 10, "error_count": 0, "avg_score": 4.0},
        ])
        actions = await evaluate_tools(memory)
        assert len(actions) == 1
        assert actions[0]["action"] == "disable"

    @pytest.mark.asyncio
    async def test_promote_high_score(self, memory):
        memory.get_generated_tools = AsyncMock(return_value=[
            {"name": "great_tool", "usage_count": 15, "error_count": 0, "avg_score": 8.5},
        ])
        actions = await evaluate_tools(memory)
        assert len(actions) == 1
        assert actions[0]["action"] == "promote"

    @pytest.mark.asyncio
    async def test_no_action_insufficient_usage(self, memory):
        memory.get_generated_tools = AsyncMock(return_value=[
            {"name": "new_tool", "usage_count": 2, "error_count": 1, "avg_score": 3.0},
        ])
        actions = await evaluate_tools(memory)
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_suggest_regenerate(self, memory):
        memory.get_generated_tools = AsyncMock(return_value=[
            {"name": "meh_tool", "usage_count": 10, "error_count": 2, "avg_score": 6.0},
        ])
        actions = await evaluate_tools(memory)
        assert len(actions) == 1
        assert actions[0]["action"] == "regenerate"

    @pytest.mark.asyncio
    async def test_unavailable_memory(self):
        m = AsyncMock()
        m.available = False
        actions = await evaluate_tools(m)
        assert actions == []
