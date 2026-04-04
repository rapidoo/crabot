"""Tests for the Planner — unit (mocked LLM) and integration (real Ollama)."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agent.config import Settings
from agent.core.planner import Planner, _extract_json, _parse_plan
from agent.infra.retry import MaxRetriesExceeded
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.schemas import Plan


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"goal": "test", "steps": []}'
        assert _extract_json(raw) == raw

    def test_markdown_fenced(self):
        raw = '```json\n{"goal": "test", "steps": []}\n```'
        result = _extract_json(raw)
        assert result.startswith("{")
        assert json.loads(result)["goal"] == "test"

    def test_markdown_fenced_no_lang(self):
        raw = '```\n{"goal": "test", "steps": []}\n```'
        result = _extract_json(raw)
        assert json.loads(result)["goal"] == "test"

    def test_embedded_in_prose(self):
        raw = 'Here is the plan:\n{"goal": "test", "steps": []}\nDone.'
        result = _extract_json(raw)
        assert json.loads(result)["goal"] == "test"


class TestParsePlan:
    def test_valid(self):
        raw = json.dumps({
            "goal": "Write fibonacci",
            "steps": [
                {"id": 1, "tool": "code", "input": "write fib", "expected_output": "function"}
            ],
        })
        plan = _parse_plan(raw)
        assert isinstance(plan, Plan)
        assert plan.goal == "Write fibonacci"

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_plan("not json at all")

    def test_invalid_schema(self):
        raw = json.dumps({"goal": "test"})  # missing steps
        with pytest.raises(ValidationError):
            _parse_plan(raw)


# ---------------------------------------------------------------------------
# Unit tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestPlannerUnit:
    def _make_planner(self, client: OllamaClient) -> Planner:
        return Planner(client=client, router=ModelRouter(Settings()))

    def _mock_response(self, content: str) -> ChatResponse:
        return ChatResponse(content=content, model="gemma4:26b")

    @pytest.mark.asyncio
    async def test_valid_plan(self):
        plan_json = json.dumps({
            "goal": "Sort a list",
            "steps": [
                {"id": 1, "tool": "code", "input": "implement sort", "expected_output": "sorted list"}
            ],
        })
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=self._mock_response(plan_json))

        planner = self._make_planner(client)
        plan = await planner.plan("Sort a list using merge sort")

        assert isinstance(plan, Plan)
        assert plan.goal == "Sort a list"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_plan_from_markdown_fenced(self):
        plan_json = json.dumps({
            "goal": "Test",
            "steps": [{"id": 1, "tool": "none", "input": "x", "expected_output": "y"}],
        })
        content = f"```json\n{plan_json}\n```"
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=self._mock_response(content))

        planner = self._make_planner(client)
        plan = await planner.plan("Do something")
        assert plan.goal == "Test"

    @pytest.mark.asyncio
    async def test_retry_on_bad_json_then_success(self):
        good_json = json.dumps({
            "goal": "OK",
            "steps": [{"id": 1, "tool": "none", "input": "x", "expected_output": "y"}],
        })
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            side_effect=[
                self._mock_response("garbage not json"),
                self._mock_response(good_json),
            ]
        )

        planner = self._make_planner(client)
        plan = await planner.plan("Test retry")
        assert plan.goal == "OK"
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=self._mock_response("always garbage")
        )

        planner = self._make_planner(client)
        with pytest.raises(MaxRetriesExceeded):
            await planner.plan("This will fail")

    @pytest.mark.asyncio
    async def test_plan_with_context(self):
        plan_json = json.dumps({
            "goal": "With context",
            "steps": [{"id": 1, "tool": "none", "input": "x", "expected_output": "y"}],
        })
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=self._mock_response(plan_json))

        planner = self._make_planner(client)
        plan = await planner.plan("Do X", context="Previous episode: did Y with score 8.0")

        # Verify context was included in the messages
        call_args = client.chat.call_args
        messages = call_args[0][1]  # second positional arg
        user_msg = messages[-1]["content"]
        assert "Memory context" in user_msg
        assert "Previous episode" in user_msg

    @pytest.mark.asyncio
    async def test_multi_step_plan(self):
        plan_json = json.dumps({
            "goal": "Multi-step",
            "steps": [
                {"id": 1, "tool": "search", "input": "find info", "expected_output": "data"},
                {"id": 2, "tool": "code", "input": "process", "expected_output": "result"},
                {"id": 3, "tool": "file", "input": "save", "expected_output": "saved"},
            ],
            "parallel": [1, 2],
        })
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(return_value=self._mock_response(plan_json))

        planner = self._make_planner(client)
        plan = await planner.plan("Complex task")
        assert len(plan.steps) == 3
        assert plan.parallel == [1, 2]


# ---------------------------------------------------------------------------
# Integration tests (require Ollama + gemma4:26b or gemma4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPlannerIntegration:
    def _make_planner(self) -> Planner:
        settings = Settings()
        # Use gemma4 (latest) which is available locally
        settings.models.planner = "gemma4"
        return Planner(
            client=OllamaClient(base_url=settings.models.ollama_base_url),
            router=ModelRouter(settings),
        )

    @pytest.mark.asyncio
    async def test_real_plan_code_task(self):
        planner = self._make_planner()
        plan = await planner.plan("Write a Python function to compute Fibonacci numbers")
        assert isinstance(plan, Plan)
        assert len(plan.goal) > 0
        assert len(plan.steps) >= 1
        for step in plan.steps:
            assert step.tool in ("search", "code", "memory", "file", "none")

    @pytest.mark.asyncio
    async def test_real_plan_research_task(self):
        planner = self._make_planner()
        plan = await planner.plan("Find information about Python asyncio best practices")
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_real_plan_simple_question(self):
        planner = self._make_planner()
        plan = await planner.plan("What is the capital of France?")
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_real_plan_with_context(self):
        planner = self._make_planner()
        plan = await planner.plan(
            "Implement merge sort",
            context="Previous episode: user asked about sorting algorithms, score 8.5",
        )
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_plan_reliability(self):
        """Run 5 plans and count parse success rate — target >= 4/5."""
        planner = self._make_planner()
        prompts = [
            "Write a Python decorator for caching",
            "Explain how HTTP works",
            "Create a REST API endpoint for user login",
            "Read a CSV file and compute statistics",
            "Refactor this code to use async/await",
        ]
        successes = 0
        for prompt in prompts:
            try:
                plan = await planner.plan(prompt)
                if isinstance(plan, Plan) and len(plan.steps) >= 1:
                    successes += 1
            except Exception:
                pass

        assert successes >= 4, f"Only {successes}/5 plans parsed successfully"
