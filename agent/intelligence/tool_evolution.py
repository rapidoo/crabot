"""Tool evolution — lifecycle management for dynamically generated tools.

Tracks usage statistics (success, errors, scores) and auto-disables
underperforming tools or promotes high-performing ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools.registry import disable_tool

if TYPE_CHECKING:
    from agent.memory.neo4j_client import MemoryClient

logger = logging.getLogger(__name__)

# Thresholds for automatic actions
MIN_USES_FOR_DISABLE = 5
MIN_USES_FOR_PROMOTE = 10
DISABLE_SCORE_THRESHOLD = 5.0
PROMOTE_SCORE_THRESHOLD = 8.0
MAX_ERROR_RATE = 0.3


async def track_tool_usage(
    memory: MemoryClient,
    tool_name: str,
    success: bool,
    score: float = 0.0,
) -> None:
    """Record a usage event for a generated tool."""
    if not memory.available:
        return
    try:
        await memory.update_tool_stats(tool_name, success=success, score=score)
    except Exception as exc:
        logger.warning("Failed to track tool usage for %s: %s", tool_name, exc)


async def evaluate_tools(memory: MemoryClient) -> list[dict[str, Any]]:
    """Evaluate all generated tools and return recommended actions.

    Returns a list of dicts: {tool, action, reason}
    where action is "disable" | "promote" | "regenerate"
    """
    if not memory.available:
        return []

    actions: list[dict[str, Any]] = []
    try:
        tools = await memory.get_generated_tools()
    except Exception as exc:
        logger.warning("Failed to get generated tools: %s", exc)
        return []

    for tool in tools:
        name = tool.get("name", "")
        usage = tool.get("usage_count", 0)
        errors = tool.get("error_count", 0)
        avg_score = tool.get("avg_score", 0.0)

        if usage < MIN_USES_FOR_DISABLE:
            continue

        error_rate = errors / usage if usage > 0 else 0

        # Auto-disable: high error rate or very low score
        if error_rate > MAX_ERROR_RATE:
            disable_tool(name)
            actions.append({
                "tool": name,
                "action": "disable",
                "reason": f"Error rate {error_rate:.0%} exceeds {MAX_ERROR_RATE:.0%}",
            })
            logger.info("Tool %s auto-disabled: error rate %.0f%%", name, error_rate * 100)

        elif avg_score < DISABLE_SCORE_THRESHOLD:
            disable_tool(name)
            actions.append({
                "tool": name,
                "action": "disable",
                "reason": f"Avg score {avg_score:.1f} below {DISABLE_SCORE_THRESHOLD}",
            })
            logger.info("Tool %s auto-disabled: avg score %.1f", name, avg_score)

        # Promote: high score and enough usage
        elif usage >= MIN_USES_FOR_PROMOTE and avg_score >= PROMOTE_SCORE_THRESHOLD:
            actions.append({
                "tool": name,
                "action": "promote",
                "reason": f"Avg score {avg_score:.1f} after {usage} uses",
            })
            logger.info("Tool %s promoted: avg score %.1f, %d uses", name, avg_score, usage)

        # Suggest regeneration: mediocre but not terrible
        elif error_rate > 0.15 and avg_score < 7.0:
            actions.append({
                "tool": name,
                "action": "regenerate",
                "reason": f"Avg score {avg_score:.1f}, error rate {error_rate:.0%} — could improve",
            })

    return actions
