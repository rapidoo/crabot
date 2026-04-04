"""Tests for pydantic schema validation."""

import pytest
from pydantic import ValidationError

from agent.schemas import (
    Step, Plan, StepResult, CriticScore, ScoredResult, AgentResult,
)


class TestStep:
    def test_valid(self):
        s = Step(id=1, tool="code", input="print(1)", expected_output="1")
        assert s.tool == "code"

    def test_all_tool_types(self):
        for tool in ("search", "web_search", "code", "memory", "file", "none"):
            s = Step(id=1, tool=tool, input="x", expected_output="y")
            assert s.tool == tool

    def test_invalid_tool(self):
        with pytest.raises(ValidationError):
            Step(id=1, tool="invalid", input="x", expected_output="y")

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            Step(id=1, tool="code", input="x")  # missing expected_output


class TestPlan:
    def test_valid(self):
        p = Plan(
            goal="Test goal",
            steps=[Step(id=1, tool="none", input="x", expected_output="y")],
        )
        assert p.goal == "Test goal"
        assert p.parallel == []

    def test_with_parallel(self):
        p = Plan(
            goal="Test",
            steps=[
                Step(id=1, tool="code", input="a", expected_output="b"),
                Step(id=2, tool="file", input="c", expected_output="d"),
            ],
            parallel=[1, 2],
        )
        assert p.parallel == [1, 2]

    def test_empty_steps_valid(self):
        p = Plan(goal="Empty", steps=[])
        assert len(p.steps) == 0

    def test_from_json(self):
        data = {
            "goal": "Do something",
            "steps": [
                {"id": 1, "tool": "search", "input": "query", "expected_output": "results"}
            ],
        }
        p = Plan.model_validate(data)
        assert p.steps[0].tool == "search"


class TestCriticScore:
    def test_valid(self):
        cs = CriticScore(
            scores={"completeness": 8, "accuracy": 9, "format": 7, "coherence": 8},
            final_score=8.0,
            retry=False,
        )
        assert cs.final_score == 8.0

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            CriticScore(
                scores={}, final_score=11.0, retry=False
            )

    def test_with_retry_reason(self):
        cs = CriticScore(
            scores={"completeness": 3},
            final_score=3.0,
            retry=True,
            reason="Output is incomplete",
        )
        assert cs.retry is True
        assert cs.reason == "Output is incomplete"


class TestStepResult:
    def test_valid(self):
        sr = StepResult(step_id=1, output="result", tokens_used=100)
        assert sr.tokens_used == 100

    def test_defaults(self):
        sr = StepResult(step_id=1, output="result")
        assert sr.tokens_used == 0


class TestScoredResult:
    def test_valid(self):
        step = Step(id=1, tool="none", input="x", expected_output="y")
        result = StepResult(step_id=1, output="y")
        score = CriticScore(scores={}, final_score=8.0, retry=False)
        sr = ScoredResult(step=step, result=result, score=score)
        assert sr.score.final_score == 8.0


class TestAgentResult:
    def test_valid(self):
        step = Step(id=1, tool="none", input="x", expected_output="y")
        result = StepResult(step_id=1, output="y")
        score = CriticScore(scores={}, final_score=8.0, retry=False)
        ar = AgentResult(
            goal="Test goal",
            results=[ScoredResult(step=step, result=result, score=score)],
        )
        assert ar.goal == "Test goal"
        assert len(ar.results) == 1
