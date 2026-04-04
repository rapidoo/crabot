"""Tests for the Executor — unit (mocked) and integration."""

from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.schemas import Plan, Step, StepResult
from agent.core.executor import Executor
from agent.tools.registry import register_tool, clear_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTool:
    def __init__(self, name: str, response: str = "tool_output"):
        self._name = name
        self._response = response

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock {self._name}"

    async def run(self, input: str) -> str:
        return self._response


def _make_plan(*steps_data: tuple[int, str, str, str]) -> Plan:
    steps = [
        Step(id=sid, tool=tool, input=inp, expected_output=exp)
        for sid, tool, inp, exp in steps_data
    ]
    return Plan(goal="Test", steps=steps)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestExecutorUnit:
    def setup_method(self):
        clear_registry()
        register_tool("code", lambda: MockTool("code", "code_result"))
        register_tool("file", lambda: MockTool("file", "file_result"))

    @pytest.mark.asyncio
    async def test_execute_tool_step(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = _make_plan((1, "code", "print(1)", "1"))
        results = await executor.execute(plan)

        assert len(results) == 1
        assert results[0].output == "code_result"
        assert results[0].step_id == 1

    @pytest.mark.asyncio
    async def test_execute_llm_step(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content="llm_output", eval_count=50)
        )
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = _make_plan((1, "none", "generate text", "text output"))
        results = await executor.execute(plan)

        assert len(results) == 1
        assert results[0].output == "llm_output"
        assert results[0].tokens_used == 50

    @pytest.mark.asyncio
    async def test_multi_step_sequential(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content="llm_out", eval_count=10)
        )
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = _make_plan(
            (1, "code", "run code", "result"),
            (2, "file", "read file", "content"),
            (3, "none", "generate", "output"),
        )
        results = await executor.execute(plan)

        assert len(results) == 3
        assert results[0].output == "code_result"
        assert results[1].output == "file_result"
        assert results[2].output == "llm_out"

    @pytest.mark.asyncio
    async def test_on_step_done_callback(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = _make_plan(
            (1, "code", "a", "b"),
            (2, "file", "c", "d"),
        )
        completed_ids: list[int] = []

        async def on_done(step_id: int):
            completed_ids.append(step_id)

        await executor.execute(plan, on_step_done=on_done)
        assert completed_ids == [1, 2]

    @pytest.mark.asyncio
    async def test_unknown_tool_falls_back_to_llm(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content="fallback", eval_count=10)
        )
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = _make_plan((1, "search", "find something", "results"))
        results = await executor.execute(plan)

        assert results[0].output == "fallback"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_direct(self):
        client = AsyncMock(spec=OllamaClient)
        client.chat = AsyncMock(
            return_value=ChatResponse(content="direct answer", eval_count=20)
        )
        executor = Executor(client=client, router=ModelRouter(Settings()))

        result = await executor.execute_direct("What is 2+2?")
        assert result.output == "direct answer"
        assert result.step_id == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExecutorIntegration:
    def setup_method(self):
        # Re-register real tools
        clear_registry()
        import agent.tools.file_io  # noqa: F401
        import agent.tools.code_exec  # noqa: F401

    @pytest.mark.asyncio
    async def test_real_code_exec(self):
        settings = Settings()
        settings.models.executor = "gemma4"
        executor = Executor(
            client=OllamaClient(base_url=settings.models.ollama_base_url),
            router=ModelRouter(settings),
        )

        plan = _make_plan((1, "code", "print(2 ** 10)", "1024"))
        results = await executor.execute(plan)
        assert "1024" in results[0].output

    @pytest.mark.asyncio
    async def test_real_mixed_plan(self):
        settings = Settings()
        settings.models.executor = "gemma4"
        executor = Executor(
            client=OllamaClient(base_url=settings.models.ollama_base_url),
            router=ModelRouter(settings),
        )

        plan = _make_plan(
            (1, "code", "print('hello from code')", "hello from code"),
            (2, "none", "Say 'world' in one word only", "world"),
        )
        results = await executor.execute(plan)
        assert len(results) == 2
        assert "hello from code" in results[0].output
        assert len(results[1].output) > 0
