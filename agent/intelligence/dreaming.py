"""Dreaming — bio-inspired memory consolidation.

Scores entities on 4 signals: frequency, relevance, diversity, recency.
Promotes high-scoring entities by reinforcing their connections.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Scoring weights (inspired by OpenClaw)
W_FREQUENCY = 0.35
W_RELEVANCE = 0.35
W_DIVERSITY = 0.15
W_RECENCY = 0.15

HALF_LIFE_DAYS = 30
MIN_SCORE_TO_PROMOTE = 0.6


async def run_dreaming(memory: object, days: int = 7) -> dict:
    """Run a dreaming cycle over recent episodes.

    Analyzes entities from the last N days and reinforces important ones.
    Returns stats about the consolidation.
    """
    if not getattr(memory, 'available', False):
        return {"status": "skipped", "reason": "memory unavailable"}

    try:
        # Get recent episodes and their entities
        stats = await _consolidate(memory, days)
        logger.info(
            "Dreaming: analyzed %d entities, promoted %d",
            stats.get("analyzed", 0),
            stats.get("promoted", 0),
        )
        return stats

    except Exception as exc:
        logger.warning("Dreaming failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _consolidate(memory: object, days: int) -> dict:
    """Core consolidation logic."""
    driver = getattr(memory, '_driver', None)
    if driver is None:
        return {"status": "skipped", "reason": "no driver"}

    async with driver.session() as session:
        # Get entity usage stats from recent episodes
        result = await session.run(
            """
            MATCH (ep:Episode)-[:HAS_ENTITY]->(e:Entity)
            WHERE ep.timestamp > datetime() - duration({days: $days})
            WITH e,
                 count(DISTINCT ep) AS frequency,
                 avg(ep.score) AS avg_relevance,
                 count(DISTINCT ep.goal) AS diversity
            RETURN e.name AS name,
                   e.type AS type,
                   frequency,
                   avg_relevance,
                   diversity
            ORDER BY frequency DESC
            """,
            days=days,
        )
        entities = [dict(r) async for r in result]

    if not entities:
        return {"status": "ok", "analyzed": 0, "promoted": 0}

    # Score each entity
    max_freq = max(e["frequency"] for e in entities) or 1
    max_div = max(e["diversity"] for e in entities) or 1
    promoted = 0

    for entity in entities:
        freq_norm = entity["frequency"] / max_freq
        rel_norm = (entity["avg_relevance"] or 0) / 10.0
        div_norm = entity["diversity"] / max_div
        recency = 1.0  # All within the window, so max recency

        score = (
            W_FREQUENCY * freq_norm
            + W_RELEVANCE * rel_norm
            + W_DIVERSITY * div_norm
            + W_RECENCY * recency
        )

        if score >= MIN_SCORE_TO_PROMOTE:
            # Reinforce: increase entity weight in the graph
            async with driver.session() as session:
                await session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    SET e.dream_score = $score,
                        e.last_dreamed = datetime(),
                        e.dream_count = coalesce(e.dream_count, 0) + 1
                    """,
                    name=entity["name"],
                    score=round(score, 3),
                )
            promoted += 1

    return {
        "status": "ok",
        "analyzed": len(entities),
        "promoted": promoted,
    }
