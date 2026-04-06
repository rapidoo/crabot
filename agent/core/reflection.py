"""Reflection cycle — the agent analyzes its own performance and suggests improvements."""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.infra.metrics import MetricsCollector
    from agent.intelligence.action_applier import ActionApplier
    from agent.memory.neo4j_client import MemoryClient

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """\
You are performing a self-reflection on your recent performance as an AI agent.
Analyze these metrics, errors, and step details to diagnose problems and suggest fixes.

{metrics_block}

{error_block}

{timing_block}

Based on this data, respond with JSON ONLY:
{{
  "insights": ["observation 1", "observation 2", ...],
  "actions": [
    {{"type": "adjust_threshold", "task_type": "...", "new_value": 6.0}},
    {{"type": "escalate_model", "task_type": "...", "from": "e4b", "to": "26b"}},
    {{"type": "create_skill", "name": "...", "tool_chain": ["search", "code"]}},
    {{"type": "disable_tool", "tool": "...", "reason": "..."}},
    {{"type": "modify_source", "target": "agent/prompts/critic.md", "new_value": "...", "reason": "..."}}
  ]
}}

Focus on diagnosing WHY errors happen and HOW to prevent them.
Only suggest actions backed by data. Be specific."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_error_block(metrics: MetricsCollector, last_n: int) -> str:
    """Extract recent errors and low-score details from traces."""
    traces = metrics.load_last(last_n)
    error_entries = [t for t in traces if t.get("has_errors")]
    low_entries = [t for t in traces if t.get("has_low_scores")]

    if not error_entries and not low_entries:
        return "No errors or low scores in recent episodes."

    lines = ["Recent problems:"]
    for t in error_entries[-5:]:
        for step in t.get("step_details", []):
            if "error" in step:
                lines.append(
                    f"  ERROR [{step.get('tool', '?')}]: {step['error'][:200]}"
                )
                lines.append(f"    Input: {step.get('input', '?')[:150]}")
    for t in low_entries[-5:]:
        for step in t.get("step_details", []):
            if step.get("score", 10) < 6.5:
                breakdown = step.get("scores_breakdown", {})
                weak = [f"{k}={v}" for k, v in breakdown.items() if v < 6]
                lines.append(
                    f"  LOW [{step.get('tool', '?')}] score={step['score']:.0f} "
                    f"weak: {', '.join(weak) or 'none'}"
                )
                if step.get("reason"):
                    lines.append(f"    Reason: {step['reason'][:150]}")
    return "\n".join(lines)


def _build_timing_block(metrics: MetricsCollector, last_n: int) -> str:
    """Aggregate phase timing stats from traces."""
    traces = metrics.load_last(last_n)
    timings: dict[str, list[float]] = {}
    for t in traces:
        for phase, dur in t.get("timings", {}).items():
            timings.setdefault(phase, []).append(dur)

    if not timings:
        return "No timing data available."

    lines = ["Phase timing averages:"]
    for phase, durs in sorted(timings.items()):
        avg = sum(durs) / len(durs)
        mx = max(durs)
        lines.append(f"  {phase}: avg {avg:.1f}s, max {mx:.1f}s ({len(durs)} samples)")
    return "\n".join(lines)


async def reflect(
    agent: Agent,
    metrics: MetricsCollector,
    memory: MemoryClient | None = None,
    action_applier: ActionApplier | None = None,
    last_n: int = 50,
) -> dict[str, Any]:
    """Run one reflection cycle.

    Returns the parsed reflection result (insights + actions).
    Persists the reflection as a meta-episode in Neo4j if available.
    If action_applier is provided and auto_apply is enabled, applies actions.
    """
    summary = metrics.summary(last_n=last_n)

    if summary.get("total_episodes", 0) < 5:
        logger.info("Reflection: not enough data (%d episodes), skipping",
                     summary.get("total_episodes", 0))
        return {"insights": ["Not enough data yet"], "actions": []}

    metrics_block = "\n".join(f"- {k}: {v}" for k, v in summary.items())

    # Load detailed error info from recent traces
    error_block = _build_error_block(metrics, last_n)
    timing_block = _build_timing_block(metrics, last_n)

    prompt = REFLECTION_PROMPT.format(
        metrics_block=metrics_block,
        error_block=error_block,
        timing_block=timing_block,
    )

    try:
        result = await agent.run(prompt)
        raw = result.results[0].result.output if result.results else "{}"

        # Parse JSON from potentially verbose output
        text = raw.strip()
        if not text.startswith("{"):
            m = _JSON_RE.search(text)
            if m:
                text = m.group(0)
        reflection = json.loads(text)

    except (json.JSONDecodeError, IndexError, Exception) as exc:
        logger.warning("Reflection parse failed: %s", exc)
        reflection = {"insights": [f"Reflection failed: {exc}"], "actions": []}

    # Log the reflection
    logger.info(
        "Reflection: %d insights, %d actions",
        len(reflection.get("insights", [])),
        len(reflection.get("actions", [])),
    )
    for insight in reflection.get("insights", []):
        logger.info("  Insight: %s", insight)
    for action in reflection.get("actions", []):
        logger.info("  Action: %s", action)

    # Apply actions if auto_apply is enabled
    actions = reflection.get("actions", [])
    if action_applier and actions:
        mutations = await action_applier.apply(actions)
        if mutations:
            logger.info("Reflection: %d mutations applied", len(mutations))
            for m in mutations:
                logger.info("  Applied: %s on %s (%s → %s)",
                            m.action_type, m.target, m.previous_value, m.new_value)

    # Persist as meta-episode
    if memory and memory.available:
        try:
            await memory.persist_episode(
                goal="self-reflection",
                summary=f"Reflection {date.today()}: {len(reflection.get('insights', []))} insights, "
                        f"{len(reflection.get('actions', []))} actions proposed",
                score=8.0,
                entities=[{"name": "self-reflection", "type": "concept",
                           "description": f"Performance analysis {date.today()}"}],
            )
        except Exception as exc:
            logger.warning("Reflection memory write failed: %s", exc)

    return reflection
