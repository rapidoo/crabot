"""Scheduler — cron triggers, file watchers, and self-scheduling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent

logger = logging.getLogger(__name__)

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


@dataclass
class ScheduledTask:
    """A single scheduled trigger."""
    name: str
    prompt: str
    schedule_type: str  # "cron" | "interval" | "once"
    schedule_value: str  # cron expr, seconds, or ISO timestamp
    next_run: datetime = field(default_factory=datetime.now)
    enabled: bool = True

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        return self.enabled and now >= self.next_run

    def advance(self) -> None:
        """Compute next_run based on schedule type."""
        if self.schedule_type == "once":
            self.enabled = False
        elif self.schedule_type == "interval":
            seconds = int(self.schedule_value)
            # Anchor to scheduled time, not wall clock (anti-drift)
            self.next_run = self.next_run + timedelta(seconds=seconds)
        elif self.schedule_type == "cron" and HAS_CRONITER:
            cron = croniter(self.schedule_value, self.next_run)
            self.next_run = cron.get_next(datetime)
        else:
            # Fallback: 1 hour
            self.next_run = self.next_run + timedelta(hours=1)


@dataclass
class Event:
    """An event to be processed by the agent."""
    prompt: str
    source: str  # "user" | "cron" | "watch" | "self"


class AgentScheduler:
    """Triggers proactifs — dispatches events to the agent."""

    def __init__(self, agent: Agent):
        self._agent = agent
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._tasks: list[ScheduledTask] = []
        self._running = False

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def add_task(self, task: ScheduledTask) -> None:
        self._tasks.append(task)
        logger.info("Scheduler: added task '%s' (%s)", task.name, task.schedule_type)

    def schedule_self(self, prompt: str, delay_seconds: int, name: str = "self") -> None:
        """The agent schedules its own next action."""
        task = ScheduledTask(
            name=name,
            prompt=prompt,
            schedule_type="once",
            schedule_value="",
            next_run=datetime.now() + timedelta(seconds=delay_seconds),
        )
        self.add_task(task)
        logger.info("Scheduler: self-scheduled '%s' in %ds", name, delay_seconds)

    async def push_event(self, event: Event) -> None:
        """Push an external event (e.g. from Telegram)."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Main event loop — run cron checks and dispatch events."""
        self._running = True
        logger.info("Scheduler started with %d tasks", len(self._tasks))
        asyncio.create_task(self._cron_loop())
        asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False

    async def _cron_loop(self) -> None:
        """Poll scheduled tasks every 30 seconds."""
        while self._running:
            now = datetime.now()
            for task in self._tasks:
                if task.is_due(now):
                    logger.info("Scheduler: triggering '%s'", task.name)
                    await self._queue.put(Event(prompt=task.prompt, source="cron"))
                    task.advance()
            # Prune disabled one-shot tasks
            self._tasks = [t for t in self._tasks if t.enabled]
            await asyncio.sleep(30)

    async def _dispatch_loop(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            try:
                logger.info("Scheduler: dispatching event from '%s'", event.source)
                await self._agent.run(event.prompt)
            except Exception as exc:
                logger.error("Scheduler: event processing failed: %s", exc)
