"""Tests for the Agent orchestrator — unit and integration."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent.config import Settings
from agent.agent import Agent
from agent.schemas import (
    Plan, Step, StepResult, CriticScore, ScoredResult, AgentResult,
)


# ---------------------------------------------------------------------------
# Unit tests (everything mocked)
# ---------------------------------------------------------------------------


class TestAgentUnit:
    @pytest.mark.asyncio
    async def test_pipeline_order(self):
        """Verify Triage → Plan → Execute → Critique ordering."""
        call_order: list[str] = []

        plan = Plan(
            goal="Test",
            steps=[Step(id=1, tool="none", input="x", expected_output="y")],
        )
        step_result = StepResult(step_id=1, output="y")
        critic_score = CriticScore(
            scores={}, final_score=8.0, retry=False
        )
        scored = [ScoredResult(step=plan.steps[0], result=step_result, score=critic_score)]

        settings = Settings()
        settings.recovery.state_file = "/tmp/nano_test_state.json"
        settings.logging.trace_file = "/tmp/nano_test_trace.jsonl"
        agent = Agent(settings)

        # Patch internal components
        agent._triage.classify = AsyncMock(
            side_effect=lambda *a, **kw: (call_order.append("triage"), "complex")[1]
        )
        agent._planner.plan = AsyncMock(
            side_effect=lambda *a, **kw: (call_order.append("plan"), plan)[1]
        )
        agent._executor.execute = AsyncMock(
            side_effect=lambda *a, **kw: (call_order.append("execute"), [step_result])[1]
        )
        agent._critic.evaluate = AsyncMock(
            side_effect=lambda *a, **kw: (call_order.append("critique"), scored)[1]
        )

        result = await agent.run("Test input")

        assert call_order == ["triage", "plan", "execute", "critique"]
        assert isinstance(result, AgentResult)
        assert result.goal == "Test"
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_simple_triage_bypasses_pipeline(self):
        """Simple inputs bypass Plan/Execute/Critique."""
        settings = Settings()
        settings.recovery.state_file = "/tmp/nano_test_state.json"
        settings.logging.trace_file = "/tmp/nano_test_trace.jsonl"
        agent = Agent(settings)

        agent._triage.classify = AsyncMock(return_value="simple")
        agent._executor.execute_direct = AsyncMock(
            return_value=StepResult(step_id=0, output="42", tokens_used=5)
        )
        # These should NOT be called
        agent._planner.plan = AsyncMock()
        agent._critic.evaluate = AsyncMock()

        result = await agent.run("What is 2+2?")

        assert result.results[0].result.output == "42"
        agent._planner.plan.assert_not_called()
        agent._critic.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_structure(self):
        plan = Plan(
            goal="Goal",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="b"),
                Step(id=2, tool="none", input="c", expected_output="d"),
            ],
        )
        results = [
            StepResult(step_id=1, output="b"),
            StepResult(step_id=2, output="d"),
        ]
        scored = [
            ScoredResult(
                step=plan.steps[i],
                result=results[i],
                score=CriticScore(scores={}, final_score=8.0, retry=False),
            )
            for i in range(2)
        ]

        settings = Settings()
        settings.recovery.state_file = "/tmp/nano_test_state.json"
        settings.logging.trace_file = "/tmp/nano_test_trace.jsonl"
        agent = Agent(settings)
        agent._triage.classify = AsyncMock(return_value="complex")
        agent._planner.plan = AsyncMock(return_value=plan)
        agent._executor.execute = AsyncMock(return_value=results)
        agent._critic.evaluate = AsyncMock(return_value=scored)

        result = await agent.run("Do two things")
        assert len(result.results) == 2
        assert result.results[0].step.id == 1
        assert result.results[1].step.id == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        settings = Settings()
        settings.models.triage = "gemma4"
        settings.models.planner = "gemma4"
        settings.models.executor = "gemma4"
        settings.models.critic_light = "gemma4"
        settings.logging.trace_file = "/tmp/nano_test_trace.jsonl"
        settings.recovery.state_file = "/tmp/nano_test_state.json"

        agent = Agent(settings)
        result = await agent.run("What is the result of 3 * 7?")

        assert isinstance(result, AgentResult)
        assert len(result.goal) > 0
        assert len(result.results) >= 1

        for sr in result.results:
            assert sr.score.final_score >= 0
            assert sr.score.final_score <= 10
