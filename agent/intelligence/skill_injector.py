"""Skill injector — injects learned skills AND Anthropic-format skills into the Planner prompt."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.skills.manager import SkillManager

logger = logging.getLogger(__name__)


async def get_skill_context(
    memory: object,
    min_score: float = 7.0,
    skill_manager: "SkillManager | None" = None,
    user_input: str = "",
) -> str:
    """Build a skill context string for injection into the Planner.

    Combines two sources:
    1. Learned tool-chain patterns from Neo4j (existing behavior)
    2. Anthropic-format skills matched to the current user input

    Returns empty string if no skills or memory unavailable.
    """
    parts: list[str] = []

    # Source 1: Learned patterns from Neo4j
    neo4j_ctx = await _get_neo4j_skills(memory, min_score)
    if neo4j_ctx:
        parts.append(neo4j_ctx)

    # Source 2: Anthropic-format skills matched to user input
    if skill_manager and user_input:
        anthropic_ctx = await _get_anthropic_skills(skill_manager, user_input)
        if anthropic_ctx:
            parts.append(anthropic_ctx)

    return "\n\n".join(parts)


async def _get_neo4j_skills(memory: object, min_score: float) -> str:
    """Retrieve learned tool-chain skills from Neo4j."""
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
    logger.debug("Skill injection: %d learned skills injected", len(lines) - 1)
    return context


async def _get_anthropic_skills(
    skill_manager: "SkillManager", user_input: str
) -> str:
    """Match and format Anthropic-format skills for the planner context."""
    try:
        matched = await skill_manager.match_skills(user_input)
    except Exception as exc:
        logger.warning("Anthropic skill matching failed: %s", exc)
        return ""

    if not matched:
        return ""

    lines = ["Relevant skill instructions:"]
    for skill in matched[:3]:  # Limit to 3 skills to control context size
        name = skill.metadata.name
        # Truncate skill content to avoid overwhelming the context
        content = skill.content
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        lines.append(f"\n### Skill: {name}\n{content}")

    context = "\n".join(lines)
    logger.info(
        "Anthropic skill injection: %d skills matched for input",
        len(matched),
    )
    return context
