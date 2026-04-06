"""Tests for PromptManager — versioned prompts with A/B testing."""

from unittest.mock import AsyncMock

import pytest

from agent.intelligence.prompt_manager import PromptManager


@pytest.fixture
def memory():
    m = AsyncMock()
    m.available = True
    m.get_active_prompt = AsyncMock(return_value=None)
    m.get_candidate_prompt = AsyncMock(return_value=None)
    m.persist_prompt_version = AsyncMock()
    m.promote_prompt = AsyncMock()
    m.update_prompt_stats = AsyncMock()
    return m


@pytest.fixture
def pm(memory):
    return PromptManager(memory=memory, eval_window=4)


class TestPromptManager:
    @pytest.mark.asyncio
    async def test_fallback_to_default(self, pm):
        result = await pm.get_prompt("planner", "default prompt")
        assert result == "default prompt"

    @pytest.mark.asyncio
    async def test_returns_active_prompt(self, memory):
        memory.get_active_prompt = AsyncMock(return_value={
            "id": "p1", "content": "evolved prompt", "version": 1,
        })
        pm = PromptManager(memory=memory)
        result = await pm.get_prompt("planner", "default")
        assert result == "evolved prompt"

    @pytest.mark.asyncio
    async def test_ab_alternation(self, memory):
        memory.get_active_prompt = AsyncMock(return_value={
            "id": "p1", "content": "active", "version": 1,
        })
        memory.get_candidate_prompt = AsyncMock(return_value={
            "id": "p2", "content": "candidate", "version": 2,
        })
        pm = PromptManager(memory=memory)

        r1 = await pm.get_prompt("planner", "default")  # episode 0 → active
        r2 = await pm.get_prompt("planner", "default")  # episode 1 → candidate
        r3 = await pm.get_prompt("planner", "default")  # episode 2 → active
        r4 = await pm.get_prompt("planner", "default")  # episode 3 → candidate

        assert r1 == "active"
        assert r2 == "candidate"
        assert r3 == "active"
        assert r4 == "candidate"

    @pytest.mark.asyncio
    async def test_propose_mutation(self, pm, memory):
        prompt_id = await pm.propose_mutation("planner", "new prompt", "testing")
        assert prompt_id is not None
        memory.persist_prompt_version.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_score(self, pm, memory):
        # Set up an active prompt
        memory.get_active_prompt = AsyncMock(return_value={
            "id": "p1", "content": "active", "version": 1,
        })
        await pm.get_prompt("planner", "default")
        await pm.record_score("planner", 8.0)
        memory.update_prompt_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_promote_when_candidate_better(self, memory):
        memory.get_active_prompt = AsyncMock(return_value={
            "id": "p1", "content": "active", "version": 1,
            "score_avg": 6.0, "usage_count": 5,
        })
        memory.get_candidate_prompt = AsyncMock(return_value={
            "id": "p2", "content": "candidate", "version": 2,
            "score_avg": 7.0, "usage_count": 5,
        })

        pm = PromptManager(memory=memory, eval_window=4)
        # Simulate having the prompts cached
        pm._cache["planner"] = {
            "active_id": "p1", "active_content": "active", "active_version": 1,
            "candidate_id": "p2", "candidate_content": "candidate", "candidate_version": 2,
        }
        pm._episode_counter["planner"] = 5  # Past eval window

        promoted = await pm.evaluate_and_promote("planner")
        assert promoted is True
        memory.promote_prompt.assert_awaited_once_with("planner", "p2")

    @pytest.mark.asyncio
    async def test_no_promote_insufficient_delta(self, memory):
        memory.get_active_prompt = AsyncMock(return_value={
            "id": "p1", "content": "active", "version": 1,
            "score_avg": 7.0, "usage_count": 5,
        })
        memory.get_candidate_prompt = AsyncMock(return_value={
            "id": "p2", "content": "candidate", "version": 2,
            "score_avg": 7.2, "usage_count": 5,
        })

        pm = PromptManager(memory=memory, eval_window=4)
        pm._cache["planner"] = {
            "active_id": "p1", "active_content": "active", "active_version": 1,
            "candidate_id": "p2", "candidate_content": "candidate", "candidate_version": 2,
        }
        pm._episode_counter["planner"] = 5

        promoted = await pm.evaluate_and_promote("planner")
        assert promoted is False

    @pytest.mark.asyncio
    async def test_unavailable_memory(self):
        m = AsyncMock()
        m.available = False
        pm = PromptManager(memory=m)
        result = await pm.get_prompt("planner", "default")
        assert result == "default"
        assert not pm.available

    @pytest.mark.asyncio
    async def test_rollback_clears_candidate(self, pm, memory):
        await pm.propose_mutation("planner", "candidate", "test")
        result = await pm.rollback_prompt("planner")
        assert result is True
        assert pm._cache.get("planner", {}).get("candidate_id") is None
