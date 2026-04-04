"""Tests for heartbeat."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.heartbeat import run_heartbeat
from agent.schemas import AgentResult, ScoredResult, Step, StepResult, CriticScore


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_runs(self):
        mock_agent = AsyncMock()
        step = Step(id=1, tool="none", input="check", expected_output="ok")
        result = StepResult(step_id=1, output="All clear, HEARTBEAT_OK")
        score = CriticScore(scores={}, final_score=8.0, retry=False)
        mock_agent.run = AsyncMock(
            return_value=AgentResult(
                goal="heartbeat",
                results=[ScoredResult(step=step, result=result, score=score)],
            )
        )

        output = await run_heartbeat(mock_agent)
        # HEARTBEAT_OK means nothing to report
        assert output is None
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_with_findings(self):
        mock_agent = AsyncMock()
        step = Step(id=1, tool="none", input="check", expected_output="ok")
        result = StepResult(step_id=1, output="Found 3 errors in logs")
        score = CriticScore(scores={}, final_score=8.0, retry=False)
        mock_agent.run = AsyncMock(
            return_value=AgentResult(
                goal="heartbeat",
                results=[ScoredResult(step=step, result=result, score=score)],
            )
        )

        output = await run_heartbeat(mock_agent)
        assert output is not None
        assert "errors" in output
