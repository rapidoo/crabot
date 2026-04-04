"""Skill injector — injects learned skills into the Planner prompt."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def get_skill_context(memory: object, min_score: float = 7.0) -> str:
    """Build a skill context string for injection into the Planner.

    Retrieves top skills from Neo4j and formats them as prompt context.
    Returns empty string if no skills or memory unavailable.
    """
    if not getattr(memory, 'available', False):
        return ""

    try:
        skills = await memory.get_skills(min_score=min_score, limit=5)
    except Exception as exc:
        logger.warning("Skill injection failed: %s", exc)
        return ""

    if not skills:
        return ""

    lines = ["Learned patterns from past successes:"]
    for s in skills:
        name = s.get("name", "")
        chain = s.get("tool_chain", [])
        score = s.get("score", 0)
        uses = s.get("uses", 0)
        if chain:
            chain_str = " → ".join(chain)
            lines.append(f"- {name}: [{chain_str}] (score {score:.1f}, used {uses}x)")

    if len(lines) <= 1:
        return ""

    context = "\n".join(lines)
    logger.debug("Skill injection: %d skills injected", len(lines) - 1)
    return context
