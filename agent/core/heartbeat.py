"""Heartbeat — reads HEARTBEAT.md and executes proactive tasks."""

from __future__ import annotations

import logging

from agent.personality.loader import load_personality

logger = logging.getLogger(__name__)

HEARTBEAT_PROMPT = (
    "Read the following HEARTBEAT tasks and execute each one. "
    "If nothing needs attention, reply HEARTBEAT_OK.\n\n"
    "Tasks:\n{tasks}"
)


async def run_heartbeat(agent: object) -> str | None:
    """Execute heartbeat tasks from workspace/HEARTBEAT.md.

    Returns the agent's response, or None if no tasks.
    """
    personality = load_personality()
    tasks = personality.heartbeat

    if not tasks or not tasks.strip():
        logger.debug("Heartbeat: no tasks defined")
        return None

    # Filter out comments and empty lines
    task_lines = [
        line.strip().lstrip("- ")
        for line in tasks.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not task_lines:
        logger.debug("Heartbeat: no actionable tasks")
        return None

    logger.info("Heartbeat: executing %d tasks", len(task_lines))

    prompt = HEARTBEAT_PROMPT.format(tasks="\n".join(f"- {t}" for t in task_lines))

    try:
        result = await agent.run(prompt)  # type: ignore
        output = "\n".join(r.result.output[:200] for r in result.results)

        if "HEARTBEAT_OK" in output.upper():
            logger.info("Heartbeat: nothing to report")
            return None

        logger.info("Heartbeat: completed with findings")
        return output

    except Exception as exc:
        logger.warning("Heartbeat failed: %s", exc)
        return None
