"""Tests for strategy evolver — trace analysis and mutation suggestions."""

import json
from pathlib import Path

import pytest

from agent.intelligence.strategy_evolver import (
    analyze_triage,
    analyze_sot,
    analyze_critic,
    evolve_strategy,
)


def _make_traces(path: str, entries: list[dict]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


class TestAnalyzeTriage:
    def test_detects_misclassification(self):
        traces = [
            {"path": "simple", "scores": [4.0]},
            {"path": "simple", "scores": [5.0]},
            {"path": "simple", "scores": [3.0]},
            {"path": "simple", "scores": [4.5]},
            {"path": "simple", "scores": [8.0]},
        ]
        actions = analyze_triage(traces)
        assert len(actions) == 1
        assert actions[0]["type"] == "escalate_model" or "triage" in str(actions[0])

    def test_no_action_when_good(self):
        traces = [
            {"path": "simple", "scores": [8.0]},
            {"path": "simple", "scores": [7.5]},
            {"path": "simple", "scores": [9.0]},
            {"path": "simple", "scores": [7.0]},
            {"path": "simple", "scores": [8.5]},
        ]
        actions = analyze_triage(traces)
        assert len(actions) == 0

    def test_insufficient_data(self):
        traces = [{"path": "simple", "scores": [4.0]}]
        assert analyze_triage(traces) == []


class TestAnalyzeSoT:
    def test_suggests_lower_threshold(self):
        traces = [
            {"path": "complex", "steps": 5, "scores": [5.0, 4.0, 6.0, 5.5, 4.5]}
            for _ in range(15)
        ]
        actions = analyze_sot(traces)
        assert len(actions) == 1
        assert actions[0]["new_value"] == 300

    def test_no_action_when_good(self):
        traces = [
            {"path": "complex", "steps": 5, "scores": [8.0, 7.0, 9.0, 8.5, 7.5]}
            for _ in range(15)
        ]
        actions = analyze_sot(traces)
        assert len(actions) == 0


class TestAnalyzeCritic:
    def test_detects_leniency_bias(self):
        traces = [{"scores": [9.0, 9.5, 8.5]} for _ in range(10)]
        actions = analyze_critic(traces)
        assert len(actions) == 1
        assert actions[0]["type"] == "modify_source"
        assert "stricter" in actions[0]["reason"]

    def test_detects_harshness_bias(self):
        traces = [{"scores": [3.0, 4.0, 4.5]} for _ in range(10)]
        actions = analyze_critic(traces)
        assert len(actions) == 1
        assert actions[0]["type"] == "modify_source"
        assert "relaxing" in actions[0]["reason"]

    def test_no_bias(self):
        traces = [{"scores": [6.5, 7.0, 7.5]} for _ in range(10)]
        actions = analyze_critic(traces)
        assert len(actions) == 0

    def test_insufficient_data(self):
        traces = [{"scores": [9.0]}]
        assert analyze_critic(traces) == []


class TestEvolveStrategy:
    @pytest.mark.asyncio
    async def test_full_analysis(self, tmp_path):
        trace_file = str(tmp_path / "traces.jsonl")
        entries = [
            {"path": "simple", "scores": [4.0]},
            {"path": "simple", "scores": [3.0]},
            {"path": "simple", "scores": [5.0]},
            {"path": "simple", "scores": [4.0]},
            {"path": "simple", "scores": [8.0]},
        ]
        _make_traces(trace_file, entries)
        suggestions = await evolve_strategy(trace_file)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        trace_file = str(tmp_path / "empty.jsonl")
        Path(trace_file).touch()
        suggestions = await evolve_strategy(trace_file)
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path):
        suggestions = await evolve_strategy(str(tmp_path / "nonexistent.jsonl"))
        assert suggestions == []
