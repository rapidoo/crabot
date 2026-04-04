"""Daemon mode — runs the Telegram bot as a long-lived service."""

from __future__ import annotations

import asyncio
import logging

from agent.agent import Agent
from agent.config import Settings, get_settings
from agent.core.scheduler import AgentScheduler, ScheduledTask
from agent.env import load_dotenv
from agent.interfaces.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


async def run_daemon(settings: Settings | None = None) -> None:
    """Start the agent in daemon mode with Telegram bot."""
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

    # Start scheduler if enabled
    scheduler: AgentScheduler | None = None
    if settings.scheduler.enabled:
        scheduler = AgentScheduler(agent)
        for task_cfg in settings.scheduler.cron_tasks:
            scheduler.add_task(ScheduledTask(
                name=task_cfg.name,
                prompt=task_cfg.prompt,
                schedule_type=task_cfg.schedule_type,
                schedule_value=task_cfg.schedule,
            ))
        await scheduler.start()
        logger.info("Scheduler started with %d tasks", scheduler.task_count)

    try:
        bot = TelegramBot(agent, settings)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if scheduler:
            await scheduler.stop()
        await agent.shutdown()
        logger.info("=== Nano Agent Daemon stopped ===")
