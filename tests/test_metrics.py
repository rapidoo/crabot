"""Tests for infra/metrics.py — MetricsCollector."""

import json
from pathlib import Path

from agent.infra.metrics import MetricsCollector


def _write_traces(path: Path, traces: list[dict]) -> None:
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


def test_empty_trace_file():
    mc = MetricsCollector("/nonexistent/path.jsonl")
    assert mc.summary() == {"total_episodes": 0}
    assert mc.get_scores() == []


def test_get_scores_flat(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    traces = [
        {"scores": [7.5, 8.0], "path": "complex", "elapsed_s": 5.0},
        {"scores": [6.0], "path": "simple", "elapsed_s": 1.0},
        {"scores": [9.0, 8.5], "path": "complex", "elapsed_s": 8.0},
    ]
    _write_traces(trace_file, traces)

    mc = MetricsCollector(str(trace_file))
    all_scores = mc.get_scores()
    assert len(all_scores) == 5
    assert all_scores == [7.5, 8.0, 6.0, 9.0, 8.5]

    complex_scores = mc.get_scores(task_type="complex")
    assert len(complex_scores) == 4


def test_summary_basic(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    traces = [
        {"scores": [8.0, 7.0], "path": "complex", "elapsed_s": 10.0},
        {"scores": [9.0], "path": "simple", "elapsed_s": 1.5},
    ]
    _write_traces(trace_file, traces)

    mc = MetricsCollector(str(trace_file))
    s = mc.summary()
    assert s["total_episodes"] == 2
    assert s["avg_score"] == 8.0  # (8+7+9) / 3
    assert "complex" in s["by_task_type"]
    assert "simple" in s["by_task_type"]
    assert s["avg_latency_s"] == 5.75


def test_summary_retry_rate(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    traces = [
        {"scores": [5.0], "path": "complex", "elapsed_s": 5.0},  # below 6.5
        {"scores": [8.0], "path": "simple", "elapsed_s": 1.0},
    ]
    _write_traces(trace_file, traces)

    mc = MetricsCollector(str(trace_file))
    s = mc.summary()
    assert s["retry_rate"] == 0.5  # 1 of 2 had low scores
