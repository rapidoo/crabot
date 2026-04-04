"""Goal Engine — manages persistent goals with decomposition and execution."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run_goal_cycle(agent: object) -> int:
    """Process the highest-priority active goal.

    Returns the number of goals processed (0 or 1).
    """
    memory = getattr(agent, '_memory', None)
    if memory is None or not memory.available:
        return 0

    goals = await memory.get_active_goals()
    if not goals:
        logger.debug("Goal engine: no active goals")
        return 0

    goal = goals[0]  # Highest priority
    goal_id = goal.get("id", "")
    description = goal.get("description", "")

    logger.info("Goal engine: working on '%s' (priority %s)", description, goal.get("priority"))

    try:
        result = await agent.run(description)  # type: ignore

        # If all steps scored well, mark goal as complete
        if result.results:
            avg_score = sum(r.score.final_score for r in result.results) / len(result.results)
            if avg_score >= 6.5:
                episode_id = getattr(agent, '_last_episode_id', None)
                await memory.complete_goal(goal_id, episode_id)
                logger.info("Goal engine: completed '%s' (score %.1f)", description, avg_score)
                return 1
            else:
                logger.info("Goal engine: '%s' needs more work (score %.1f)", description, avg_score)

    except Exception as exc:
        logger.warning("Goal engine failed on '%s': %s", description, exc)

    return 0
