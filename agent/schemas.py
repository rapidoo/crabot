"""Pydantic models — shared data contracts for the agent pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Plan (Planner output)
# ---------------------------------------------------------------------------


class Step(BaseModel):
    id: int
    tool: Literal["search", "code", "memory", "file", "none"]
    input: str
    expected_output: str


class Plan(BaseModel):
    goal: str
    steps: list[Step]
    parallel: list[int] = []


# ---------------------------------------------------------------------------
# Execution (Executor output)
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    step_id: int
    output: str
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Critique (Critic output)
# ---------------------------------------------------------------------------


class CriticScore(BaseModel):
    scores: dict[str, float] = Field(
        description="Keys: completeness, accuracy, format, coherence (0-10 each)"
    )
    final_score: float = Field(ge=0, le=10)
    retry: bool
    reason: str = ""


class ScoredResult(BaseModel):
    step: Step
    result: StepResult
    score: CriticScore


# ---------------------------------------------------------------------------
# Agent (final output)
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    goal: str
    results: list[ScoredResult]
