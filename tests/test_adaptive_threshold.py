"""Tests for AdaptiveThreshold in critic.py."""

import json

from agent.core.critic import AdaptiveThreshold


def test_default_when_no_data():
    threshold = AdaptiveThreshold(default=6.5)
    assert threshold.get() == 6.5


def test_default_when_insufficient_samples(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    with open(trace_file, "w") as f:
        # Only 5 entries, min_samples=20
        for _ in range(5):
            f.write(json.dumps({"scores": [7.0], "path": "complex"}) + "\n")

    threshold = AdaptiveThreshold(default=6.5, min_samples=20, trace_file=str(trace_file))
    assert threshold.get() == 6.5


def test_adaptive_with_enough_samples(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    with open(trace_file, "w") as f:
        # 25 entries with scores around 7-9
        for i in range(25):
            score = 7.0 + (i % 3)  # 7, 8, 9, 7, 8, 9, ...
            f.write(json.dumps({"scores": [score], "path": "complex"}) + "\n")

    threshold = AdaptiveThreshold(default=6.5, min_samples=20, trace_file=str(trace_file))
    result = threshold.get()
    # mean=8.0, std~0.82 -> threshold ~ 7.2
    assert 6.0 < result < 8.5
    assert result != 6.5  # should have adapted


def test_adaptive_floor(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    with open(trace_file, "w") as f:
        # All low scores -> threshold should not go below 4.0
        for _ in range(25):
            f.write(json.dumps({"scores": [2.0], "path": "complex"}) + "\n")

    threshold = AdaptiveThreshold(default=6.5, min_samples=20, trace_file=str(trace_file))
    result = threshold.get()
    assert result == 4.0  # floor
