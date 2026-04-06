"""Lesson injector — retrieves relevant lessons and formats them for Planner context."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def get_lesson_context(
    memory: object,
    query_embedding: list[float] | None = None,
    limit: int = 5,
) -> str:
    """Build a lesson context string for injection into the Planner.

    Retrieves semantically relevant lessons from Neo4j and formats them
    as high-priority context. Returns empty string if none found.
    """
    if not getattr(memory, "available", False):
        return ""
    if query_embedding is None:
        return ""

    try:
        lessons = await memory.get_relevant_lessons(
            query_embedding=query_embedding, limit=limit
        )
    except Exception as exc:
        logger.warning("Lesson injection failed: %s", exc)
        return ""

    if not lessons:
        return ""

    lines = ["Important lessons from past corrections (follow these strictly):"]
    for lesson in lessons:
        rule = lesson.get("rule", "")
        context = lesson.get("context", "")
        category = lesson.get("category", "correction")
        reinforced = lesson.get("times_reinforced", 0)

        prefix = f"[{category}]"
        ctx_part = f" {context}:" if context else ""
        reinf_part = f" (\u00d7{reinforced + 1})" if reinforced > 0 else ""
        lines.append(f"- {prefix}{ctx_part} {rule}{reinf_part}")

    if len(lines) <= 1:
        return ""

    context = "\n".join(lines)
    logger.debug("Lesson injection: %d lessons injected", len(lines) - 1)
    return context
