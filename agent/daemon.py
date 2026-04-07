"""Daemon mode — runs the Telegram bot with heartbeat, goals, and reflection."""

from __future__ import annotations

import asyncio
import logging

from agent.agent import Agent
from agent.config import Settings, get_settings
from agent.core.heartbeat import run_heartbeat
from agent.core.goal_engine import run_goal_cycle
from agent.core.job_manager import JobManager
from agent.core.scheduler import AgentScheduler, ScheduledTask, Event
from agent.env import load_dotenv
from agent.interfaces.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class _HeartbeatScheduler(AgentScheduler):
    """Scheduler that intercepts special prompts for heartbeat/goals/reflection."""

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
                elif event.prompt == "__reflection__":
                    await self._run_reflection()
                elif event.prompt == "__workspace_evolve__":
                    await self._run_workspace_evolution()
                else:
                    logger.info("Scheduler: dispatching event from '%s'", event.source)
                    await self._agent.run(event.prompt)
            except Exception as exc:
                logger.error("Scheduler: event processing failed: %s", exc)

    async def _run_reflection(self) -> None:
        settings = self._agent._settings
        if not settings.reflection.enabled:
            return
        try:
            from agent.core.reflection import reflect
            from agent.infra.metrics import MetricsCollector
            mc = MetricsCollector(settings.logging.trace_file)
            await reflect(
                agent=self._agent,
                metrics=mc,
                memory=self._agent._memory if self._agent._memory.available else None,
                action_applier=self._agent._action_applier,
                last_n=settings.reflection.last_n_episodes,
            )
            logger.info("Reflection cycle completed")
        except Exception as exc:
            logger.warning("Reflection cycle failed: %s", exc)

    async def _run_workspace_evolution(self) -> None:
        try:
            modified = await self._agent._evolve_workspace()
            if modified:
                logger.info("Workspace evolution completed: %s", modified)
        except Exception as exc:
            logger.warning("Workspace evolution failed: %s", exc)


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

    # Job manager for async background processing
    job_manager = JobManager(
        agent=agent,
        memory=agent.memory,
        max_concurrent=settings.daemon.max_concurrent,
    )

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

    # Reflection — every hour
    scheduler.add_task(ScheduledTask(
        name="reflection",
        prompt="__reflection__",
        schedule_type="interval",
        schedule_value="3600",
    ))

    # Workspace evolution — every hour
    scheduler.add_task(ScheduledTask(
        name="workspace_evolve",
        prompt="__workspace_evolve__",
        schedule_type="interval",
        schedule_value="3600",
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
    logger.info("Scheduler started: heartbeat (30m), goals (15m), reflection (1h), workspace (1h), %d user tasks",
                len(settings.scheduler.cron_tasks))

    try:
        bot = TelegramBot(agent, settings, job_manager=job_manager)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await scheduler.stop()
        await agent.shutdown()
        logger.info("=== Nano Agent Daemon stopped ===")
