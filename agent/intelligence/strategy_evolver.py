"""Strategy evolver — analyzes agent traces to detect pipeline inefficiencies
and propose configuration mutations.

Detects patterns like:
- Triage misclassification (simple tasks with low scores)
- SoT waste (activated on short outputs, missed on long ones)
- Critic bias (scores systematically too high or too low)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.infra.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Analysis thresholds
TRIAGE_MISCLASS_THRESHOLD = 6.0  # Simple-path results below this suggest misclassification
SOT_WASTE_MIN_STEPS = 10
CRITIC_BIAS_MIN_SAMPLES = 20
CRITIC_HIGH_BIAS = 8.5   # Avg score above this suggests lenient critic
CRITIC_LOW_BIAS = 5.0    # Avg score below this suggests harsh critic


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
            "type": "escalate_model",
            "task_type": "triage",
            "to": "gemma4:26b",
            "reason": f"Triage misclassification rate {misclass_rate:.0%} — simple-path results scored below {TRIAGE_MISCLASS_THRESHOLD}",
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
    """Detect critic scoring bias and propose prompt file modifications."""
    all_scores: list[float] = []
    for t in traces:
        for s in t.get("scores", []):
            all_scores.append(s)

    if len(all_scores) < CRITIC_BIAS_MIN_SAMPLES:
        return []

    avg = sum(all_scores) / len(all_scores)
    actions: list[dict[str, Any]] = []

    # Read current critic prompt from file
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "critic.md"
    current_prompt = ""
    if prompt_path.is_file():
        current_prompt = prompt_path.read_text(encoding="utf-8")

    if avg > CRITIC_HIGH_BIAS and current_prompt:
        stricter_addition = (
            "Calibration note: Be stricter in scoring. "
            "A score of 8+ should only be given for truly excellent results. "
            "Default to 6.0 for adequate-but-unremarkable answers. "
            "Deduct points for: vague answers, missing details, incorrect assumptions."
        )
        if "Be stricter" not in current_prompt:
            actions.append({
                "type": "modify_source",
                "target": "agent/prompts/critic.md",
                "patch_mode": "append",
                "new_value": stricter_addition,
                "reason": f"Critic bias detected: avg {avg:.1f} > {CRITIC_HIGH_BIAS} — adding stricter calibration",
            })
    elif avg < CRITIC_LOW_BIAS and current_prompt:
        relaxed_addition = (
            "Calibration note: Be more generous in scoring. "
            "A score of 5.0 should be reserved for genuinely poor results. "
            "Give credit for partial correctness and reasonable attempts."
        )
        if "Be more generous" not in current_prompt:
            actions.append({
                "type": "modify_source",
                "target": "agent/prompts/critic.md",
                "patch_mode": "append",
                "new_value": relaxed_addition,
                "reason": f"Critic bias detected: avg {avg:.1f} < {CRITIC_LOW_BIAS} — relaxing calibration",
            })

    return actions


async def evolve_strategy(trace_file: str) -> list[dict[str, Any]]:
    """Run all strategy analyses and return a list of suggested mutations.

    Each suggestion is a dict with 'type', 'target', 'reason', and specifics.
    These can be passed to ActionApplier.apply() or logged for human review.
    """
    traces = MetricsCollector(trace_file).load_last(100)
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
