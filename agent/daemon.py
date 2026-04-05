"""Daemon mode — runs the Telegram bot with heartbeat, goals, and reflection."""

from __future__ import annotations

import asyncio
import logging

from agent.agent import Agent
from agent.config import Settings, get_settings
from agent.core.heartbeat import run_heartbeat
from agent.core.goal_engine import run_goal_cycle
from agent.core.scheduler import AgentScheduler, ScheduledTask, Event
from agent.env import load_dotenv
from agent.interfaces.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class _HeartbeatScheduler(AgentScheduler):
    """Scheduler that intercepts special prompts for heartbeat/goals."""

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            try:
                if event.prompt == "__heartbeat__":
                    await run_heartbeat(self._agent)
                elif event.prompt == "__goal_cycle__":
                    await run_goal_cycle(self._agent)
                else:
                    logger.info("Scheduler: dispatching event from '%s'", event.source)
                    await self._agent.run(event.prompt)
            except Exception as exc:
                logger.error("Scheduler: event processing failed: %s", exc)


async def run_daemon(settings: Settings | None = None) -> None:
    """Start the agent in daemon mode with Telegram bot + autonomous tasks."""
    load_dotenv()
    settings = settings or get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Nano Agent Daemon starting ===")

    agent = Agent(settings)
    await agent.initialize()

    # Start scheduler with built-in autonomous tasks
    scheduler = _HeartbeatScheduler(agent)

    # Heartbeat — every 30 minutes
    scheduler.add_task(ScheduledTask(
        name="heartbeat",
        prompt="__heartbeat__",
        schedule_type="interval",
        schedule_value="1800",
    ))

    # Goal cycle — every 15 minutes
    scheduler.add_task(ScheduledTask(
        name="goal_cycle",
        prompt="__goal_cycle__",
        schedule_type="interval",
        schedule_value="900",
    ))

    # User-defined cron tasks from config
    for task_cfg in settings.scheduler.cron_tasks:
        scheduler.add_task(ScheduledTask(
            name=task_cfg.name,
            prompt=task_cfg.prompt,
            schedule_type=task_cfg.schedule_type,
            schedule_value=task_cfg.schedule,
        ))

    await scheduler.start()
    logger.info("Scheduler started: heartbeat (30m), goals (15m), %d user tasks",
                len(settings.scheduler.cron_tasks))

    try:
        bot = TelegramBot(agent, settings)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await scheduler.stop()
        await agent.shutdown()
        logger.info("=== Nano Agent Daemon stopped ===")
