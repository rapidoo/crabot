"""Tests for parallel execution and Skeleton-of-Thought."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.schemas import Plan, Step
from agent.core.executor import Executor
from agent.tools.registry import register_tool, clear_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SlowTool:
    """Tool that takes a fixed delay — used to verify parallel execution."""

    def __init__(self, delay: float = 0.1, output: str = "done"):
        self._delay = delay
        self._output = output

    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "Slow tool for testing"

    async def run(self, input: str) -> str:
        await asyncio.sleep(self._delay)
        return self._output


# ---------------------------------------------------------------------------
# Parallel execution tests
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def setup_method(self):
        clear_registry()
        register_tool("code", lambda: SlowTool(0.1, "code_done"))
        register_tool("file", lambda: SlowTool(0.1, "file_done"))

    @pytest.mark.asyncio
    async def test_parallel_steps_run_concurrently(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = Plan(
            goal="Test parallel",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="x"),
                Step(id=2, tool="file", input="b", expected_output="y"),
                Step(id=3, tool="code", input="c", expected_output="z"),
            ],
            parallel=[1, 2],  # steps 1 and 2 are parallel, 3 is sequential
        )

        start = time.monotonic()
        results = await executor.execute(plan)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        # Parallel steps (0.1s each) should take ~0.1s total, not 0.2s
        # Plus sequential step 3 (~0.1s) = ~0.2s total
        # Without parallelism it would be ~0.3s
        assert elapsed < 0.35  # generous margin

    @pytest.mark.asyncio
    async def test_sequential_only(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = Plan(
            goal="Test sequential",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="x"),
                Step(id=2, tool="file", input="b", expected_output="y"),
            ],
            parallel=[],  # no parallel
        )

        results = await executor.execute(plan)
        assert len(results) == 2
        assert results[0].output == "code_done"
        assert results[1].output == "file_done"

    @pytest.mark.asyncio
    async def test_on_step_done_called_for_parallel(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = Plan(
            goal="Test callbacks",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="x"),
                Step(id=2, tool="file", input="b", expected_output="y"),
            ],
            parallel=[1, 2],
        )
        done_ids: list[int] = []

        def on_done(step_id: int) -> None:
            done_ids.append(step_id)

        await executor.execute(plan, on_step_done=on_done)
        assert sorted(done_ids) == [1, 2]

    @pytest.mark.asyncio
    async def test_resume_after_skips_steps(self):
        client = AsyncMock(spec=OllamaClient)
        executor = Executor(client=client, router=ModelRouter(Settings()))

        plan = Plan(
            goal="Test resume",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="x"),
                Step(id=2, tool="file", input="b", expected_output="y"),
                Step(id=3, tool="code", input="c", expected_output="z"),
            ],
        )

        # Resume after step 2 — only step 3 should execute
        results = await executor.execute(plan, resume_after=2)
        assert len(results) == 1
        assert results[0].step_id == 3


# ---------------------------------------------------------------------------
# Skeleton-of-Thought tests
# ---------------------------------------------------------------------------


class TestSkeletonOfThought:
    @pytest.mark.asyncio
    async def test_parse_skeleton_sections(self):
        executor = Executor(
            client=AsyncMock(spec=OllamaClient),
            router=ModelRouter(Settings()),
        )
        skeleton = """1. Introduction: Brief overview
2. Core Concepts: Key ideas
3. Implementation: Code examples
4. Conclusion: Summary"""

        sections = executor._parse_skeleton_sections(skeleton)
        assert len(sections) == 4
        assert sections[0] == "Introduction"
        assert sections[2] == "Implementation"

    @pytest.mark.asyncio
    async def test_parse_skeleton_bullet_format(self):
        executor = Executor(
            client=AsyncMock(spec=OllamaClient),
            router=ModelRouter(Settings()),
        )
        skeleton = """- Overview: What it is
- Architecture: How it works
- Usage: How to use it"""

        sections = executor._parse_skeleton_sections(skeleton)
        assert len(sections) == 3
        assert sections[0] == "Overview"

    @pytest.mark.asyncio
    async def test_sot_execution(self):
        client = AsyncMock(spec=OllamaClient)
        # First call: skeleton generation
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="1. Intro: Overview\n2. Details: Deep dive\n3. Summary: Wrap up",
                    eval_count=20,
                ),
                # Section fills (parallel)
                ChatResponse(content="Intro content here", eval_count=30),
                ChatResponse(content="Details content here", eval_count=30),
                ChatResponse(content="Summary content here", eval_count=30),
            ]
        )

        executor = Executor(client=client, router=ModelRouter(Settings()))
        step = Step(id=1, tool="none", input="Write a guide", expected_output="guide text")

        result = await executor.execute_sot(step)
        assert "## Intro" in result.output
        assert "## Details" in result.output
        assert "## Summary" in result.output
        assert "Intro content here" in result.output
