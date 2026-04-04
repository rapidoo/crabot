"""Metrics collector — aggregates JSONL traces into actionable stats."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Reads agent_trace.jsonl and computes aggregate performance stats."""

    def __init__(self, trace_file: str | Path):
        self._path = Path(trace_file)

    def _load_last(self, n: int) -> list[dict]:
        """Load the last N trace entries."""
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception as exc:
            logger.warning("Failed to load traces: %s", exc)
            return []

    def get_scores(self, task_type: str = "default", limit: int = 50) -> list[float]:
        """Flat list of final scores, optionally filtered by task type."""
        traces = self._load_last(limit)
        scores: list[float] = []
        for t in traces:
            if task_type != "default" and t.get("path") != task_type:
                continue
            for s in t.get("scores", []):
                if isinstance(s, (int, float)):
                    scores.append(float(s))
        return scores

    def summary(self, last_n: int = 100) -> dict:
        """Aggregate stats over the last N episodes."""
        traces = self._load_last(last_n)
        if not traces:
            return {"total_episodes": 0}

        all_scores: list[float] = []
        by_path: dict[str, list[float]] = defaultdict(list)
        latencies: list[float] = []
        retries = 0
        tool_counts: dict[str, int] = defaultdict(int)
        tool_scores: dict[str, list[float]] = defaultdict(list)

        for t in traces:
            scores = [s for s in t.get("scores", []) if isinstance(s, (int, float))]
            all_scores.extend(scores)
            path = t.get("path", "unknown")
            by_path[path].extend(scores)
            if "elapsed_s" in t:
                latencies.append(t["elapsed_s"])
            # Count tools used (from step data if available)
            for tool in t.get("tools_used", []):
                tool_counts[tool] += 1
            # Detect retries (episodes with scores below threshold)
            if any(s < 6.5 for s in scores):
                retries += 1

        result: dict = {
            "total_episodes": len(traces),
            "avg_score": round(mean(all_scores), 2) if all_scores else 0,
            "score_std": round(stdev(all_scores), 2) if len(all_scores) > 1 else 0,
            "by_task_type": {
                k: round(mean(v), 2) for k, v in by_path.items() if v
            },
            "avg_latency_s": round(mean(latencies), 2) if latencies else 0,
            "retry_rate": round(retries / len(traces), 2) if traces else 0,
        }

        if tool_counts:
            result["tool_usage"] = dict(tool_counts)

        return result

    def model_stats(self, task_type: str = "default") -> dict[str, float]:
        """Per-model average scores (for router learning)."""
        traces = self._load_last(200)
        by_model: dict[str, list[float]] = defaultdict(list)
        for t in traces:
            if task_type != "default" and t.get("path") != task_type:
                continue
            model = t.get("model", "unknown")
            scores = [s for s in t.get("scores", []) if isinstance(s, (int, float))]
            if scores:
                by_model[model].append(mean(scores))
        return {k: round(mean(v), 2) for k, v in by_model.items() if v}
