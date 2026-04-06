"""Strategy evolver — analyzes agent traces to detect pipeline inefficiencies
and propose configuration mutations.

Detects patterns like:
- Triage misclassification (simple tasks with low scores)
- SoT waste (activated on short outputs, missed on long ones)
- Critic bias (scores systematically too high or too low)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Analysis thresholds
TRIAGE_MISCLASS_THRESHOLD = 6.0  # Simple-path results below this suggest misclassification
SOT_WASTE_MIN_STEPS = 10
CRITIC_BIAS_MIN_SAMPLES = 20
CRITIC_HIGH_BIAS = 8.5   # Avg score above this suggests lenient critic
CRITIC_LOW_BIAS = 5.0    # Avg score below this suggests harsh critic


def _load_traces(trace_file: str, limit: int = 100) -> list[dict[str, Any]]:
    """Load the last N trace entries from the JSONL file."""
    path = Path(trace_file)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load traces: %s", exc)
    return entries[-limit:]


def analyze_triage(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect triage misclassifications: simple path with low scores."""
    simple_traces = [t for t in traces if t.get("path") == "simple"]
    if len(simple_traces) < 5:
        return []

    low_score_simple = [
        t for t in simple_traces
        if t.get("scores") and min(t["scores"]) < TRIAGE_MISCLASS_THRESHOLD
    ]
    misclass_rate = len(low_score_simple) / len(simple_traces)

    actions: list[dict[str, Any]] = []
    if misclass_rate > 0.2:
        actions.append({
            "type": "adjust_threshold",
            "target": "triage_sensitivity",
            "observation": f"{misclass_rate:.0%} of simple-path results scored below {TRIAGE_MISCLASS_THRESHOLD}",
            "suggestion": "escalate_model",
            "task_type": "triage",
            "from": "e4b",
            "to": "26b",
            "reason": f"Triage misclassification rate {misclass_rate:.0%}",
        })
    return actions


def analyze_sot(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect Skeleton-of-Thought waste or missed opportunities."""
    complex_traces = [t for t in traces if t.get("path") == "complex"]
    if len(complex_traces) < SOT_WASTE_MIN_STEPS:
        return []

    # Check if many high-step-count tasks have low scores (SoT might help)
    high_step = [t for t in complex_traces if t.get("steps", 0) > 3]
    if not high_step:
        return []

    avg_score_high = sum(
        sum(t.get("scores", [7])) / len(t.get("scores", [7]))
        for t in high_step
    ) / len(high_step)

    actions: list[dict[str, Any]] = []
    if avg_score_high < 6.5:
        actions.append({
            "type": "adjust_threshold",
            "target": "skeleton.token_threshold",
            "observation": f"Multi-step tasks avg score {avg_score_high:.1f}",
            "new_value": 300,  # Lower threshold to activate SoT more
            "reason": f"Multi-step tasks underperforming (avg {avg_score_high:.1f})",
        })
    return actions


def analyze_critic(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect critic scoring bias."""
    all_scores: list[float] = []
    for t in traces:
        for s in t.get("scores", []):
            all_scores.append(s)

    if len(all_scores) < CRITIC_BIAS_MIN_SAMPLES:
        return []

    avg = sum(all_scores) / len(all_scores)
    actions: list[dict[str, Any]] = []

    if avg > CRITIC_HIGH_BIAS:
        actions.append({
            "type": "mutate_prompt",
            "role": "critic",
            "observation": f"Critic avg score {avg:.1f} suggests leniency bias",
            "suggestion": "Add stricter evaluation criteria and lower default scores",
            "reason": f"Critic bias detected: avg {avg:.1f} > {CRITIC_HIGH_BIAS}",
        })
    elif avg < CRITIC_LOW_BIAS:
        actions.append({
            "type": "mutate_prompt",
            "role": "critic",
            "observation": f"Critic avg score {avg:.1f} suggests harshness bias",
            "suggestion": "Relax evaluation criteria slightly",
            "reason": f"Critic bias detected: avg {avg:.1f} < {CRITIC_LOW_BIAS}",
        })

    return actions


async def evolve_strategy(trace_file: str) -> list[dict[str, Any]]:
    """Run all strategy analyses and return a list of suggested mutations.

    Each suggestion is a dict with 'type', 'target', 'reason', and specifics.
    These can be passed to ActionApplier.apply() or logged for human review.
    """
    traces = _load_traces(trace_file)
    if not traces:
        logger.info("Strategy evolver: no traces to analyze")
        return []

    suggestions: list[dict[str, Any]] = []
    suggestions.extend(analyze_triage(traces))
    suggestions.extend(analyze_sot(traces))
    suggestions.extend(analyze_critic(traces))

    if suggestions:
        logger.info("Strategy evolver: %d suggestions", len(suggestions))
        for s in suggestions:
            logger.info("  %s: %s", s.get("type"), s.get("reason"))
    else:
        logger.debug("Strategy evolver: no improvements detected")

    return suggestions
