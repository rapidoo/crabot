"""Pydantic models — shared data contracts for the agent pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Plan (Planner output)
# ---------------------------------------------------------------------------


class Step(BaseModel):
    id: int
    tool: str  # Dynamic — validated against registry at runtime, not compile time
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


# ---------------------------------------------------------------------------
# Background Jobs
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    id: str
    user_input: str
    status: JobStatus = JobStatus.pending
    chat_id: int = 0
    message_id: int | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    current_phase: str = "pending"
    progress: str = ""
    result: AgentResult | None = None
    error: str | None = None
    episode_id: str | None = None


# ---------------------------------------------------------------------------
# Mutations (self-modification tracking)
# ---------------------------------------------------------------------------


class Mutation(BaseModel):
    id: str
    timestamp: str
    action_type: str  # adjust_threshold | escalate_model | disable_tool | create_skill | mutate_prompt
    target: str
    previous_value: Any = None
    new_value: Any = None
    reason: str = ""
    reflection_id: str | None = None
    rolled_back: bool = False
