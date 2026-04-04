"""Tests for the Critic — unit (mocked) and integration."""

import json
from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.schemas import Plan, Step, StepResult, CriticScore
from agent.core.critic import Critic, _parse_critic_score


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


class TestParseCriticScore:
    def test_valid_json(self):
        raw = json.dumps({
            "scores": {"completeness": 8, "accuracy": 9, "format": 7, "coherence": 8},
            "final_score": 8.0,
            "retry": False,
            "reason": "",
        })
        cs = _parse_critic_score(raw)
        assert cs.final_score == 8.0
        assert cs.retry is False

    def test_json_in_prose(self):
        raw = 'Here is my evaluation:\n' + json.dumps({
            "scores": {"completeness": 5},
            "final_score": 5.0,
            "retry": True,
            "reason": "Incomplete",
        })
        cs = _parse_critic_score(raw)
        assert cs.retry is True

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_critic_score("not json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(sid: int = 1) -> Step:
    return Step(id=sid, tool="code", input="test", expected_output="result")


def _make_result(sid: int = 1, output: str = "result") -> StepResult:
    return StepResult(step_id=sid, output=output)


def _make_plan(*step_ids: int) -> Plan:
    steps = [_make_step(sid) for sid in step_ids]
    return Plan(goal="Test", steps=steps)


def _score_response(final_score: float, retry: bool = False, reason: str = "") -> ChatResponse:
    return ChatResponse(
        content=json.dumps({
            "scores": {"completeness": final_score, "accuracy": final_score,
                       "format": final_score, "coherence": final_score},
            "final_score": final_score,
            "retry": retry,
            "reason": reason,
        })
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCriticUnit:
    @pytest.mark.asyncio
    async def test_high_score_no_retry(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=_score_response(8.0))

        critic = Critic(client=client, router=ModelRouter(Settings()))
        plan = _make_plan(1)
        results = [_make_result(1)]

        scored = await critic.evaluate(plan, results)
        assert len(scored) == 1
        assert scored[0].score.final_score == 8.0
        assert client.chat.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_low_score_triggers_retry(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            side_effect=[
                _score_response(4.0, retry=True, reason="incomplete"),
                _score_response(8.0),
            ]
        )

        executor = AsyncMock()
        executor.execute_step = AsyncMock(return_value=_make_result(1, "better"))

        critic = Critic(
            client=client, router=ModelRouter(Settings()), executor=executor
        )
        plan = _make_plan(1)
        results = [_make_result(1, "bad")]

        scored = await critic.evaluate(plan, results)
        assert scored[0].score.final_score == 8.0
        assert executor.execute_step.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=_score_response(3.0, retry=True, reason="always bad")
        )

        executor = AsyncMock()
        executor.execute_step = AsyncMock(return_value=_make_result(1, "still bad"))

        settings = Settings()
        settings.thresholds.max_retries = 3
        critic = Critic(
            client=client, router=ModelRouter(settings),
            executor=executor,
        )
        plan = _make_plan(1)
        results = [_make_result(1, "bad")]

        scored = await critic.evaluate(plan, results)
        # Should have retried max_retries-1 times (2 retries + original)
        assert scored[0].score.final_score == 3.0
        assert executor.execute_step.call_count == 2  # max_retries - 1

    @pytest.mark.asyncio
    async def test_no_executor_no_retry(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=_score_response(4.0, retry=True)
        )

        critic = Critic(client=client, router=ModelRouter(Settings()))
        plan = _make_plan(1)
        results = [_make_result(1)]

        scored = await critic.evaluate(plan, results)
        assert scored[0].score.final_score == 4.0
        assert client.chat.call_count == 1  # no retry without executor

    @pytest.mark.asyncio
    async def test_multi_step(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=_score_response(7.5))

        critic = Critic(client=client, router=ModelRouter(Settings()))
        plan = _make_plan(1, 2, 3)
        results = [_make_result(i) for i in [1, 2, 3]]

        scored = await critic.evaluate(plan, results)
        assert len(scored) == 3
        assert all(s.score.final_score == 7.5 for s in scored)

    @pytest.mark.asyncio
    async def test_parse_error_defaults_to_pass(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content="garbage not json")
        )

        critic = Critic(client=client, router=ModelRouter(Settings()))
        plan = _make_plan(1)
        results = [_make_result(1)]

        scored = await critic.evaluate(plan, results)
        assert scored[0].score.final_score == 7.0  # default pass


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCriticIntegration:
    @pytest.mark.asyncio
    async def test_real_scoring(self):
        settings = Settings()
        settings.models.critic_light = "gemma4"
        client = OllamaClient(base_url=settings.models.ollama_base_url)
        critic = Critic(client=client, router=ModelRouter(settings))

        step = Step(id=1, tool="code", input="print(2+2)", expected_output="4")
        result = StepResult(step_id=1, output="4")
        plan = Plan(goal="Compute 2+2", steps=[step])

        scored = await critic.evaluate(plan, [result])
        assert len(scored) == 1
        # Good output should score well
        assert scored[0].score.final_score >= 5.0
