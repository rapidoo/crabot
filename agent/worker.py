"""Execution worker — runs the Agent pipeline as a subprocess behind IPC.

Started by the supervisor:
    python -m agent.worker --socket /tmp/nano-agent.sock
"""

from __future__ import annotations

import asyncio
import logging
import sys
import traceback

from agent.config import get_settings, Settings
from agent.agent import Agent
from agent.core.job_manager import JobManager
from agent.core.scheduler import AgentScheduler, ScheduledTask
from agent.core.heartbeat import run_heartbeat
from agent.core.goal_engine import run_goal_cycle
from agent.env import load_dotenv
from agent.ipc import Message, send_message, recv_message

logger = logging.getLogger(__name__)


class _WorkerScheduler(AgentScheduler):
    """Scheduler with heartbeat/goal dispatch (same as daemon's)."""

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
                    logger.info("Scheduler: dispatching '%s'", event.source)
                    await self._agent.run(event.prompt)
            except Exception as exc:
                logger.error("Scheduler dispatch failed: %s", exc)


class ExecutionWorker:
    """Wraps the Agent behind a Unix socket IPC interface."""

    def __init__(self, socket_path: str, settings: Settings | None = None):
        self._socket_path = socket_path
        self._settings = settings or get_settings()
        self._agent: Agent | None = None
        self._scheduler: _WorkerScheduler | None = None
        self._running = False

    async def start(self) -> None:
        """Connect to supervisor socket, initialize agent, and serve requests."""
        self._agent = Agent(self._settings)
        await self._agent.initialize()

        # Start scheduler
        self._scheduler = _WorkerScheduler(self._agent)
        self._scheduler.add_task(ScheduledTask(
            name="heartbeat", prompt="__heartbeat__",
            schedule_type="interval", schedule_value="1800",
        ))
        self._scheduler.add_task(ScheduledTask(
            name="goal_cycle", prompt="__goal_cycle__",
            schedule_type="interval", schedule_value="900",
        ))
        for task_cfg in self._settings.scheduler.cron_tasks:
            self._scheduler.add_task(ScheduledTask(
                name=task_cfg.name, prompt=task_cfg.prompt,
                schedule_type=task_cfg.schedule_type,
                schedule_value=task_cfg.schedule,
            ))
        await self._scheduler.start()

        # Connect to supervisor
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        self._running = True

        # Announce ready
        await send_message(writer, Message.control("ready"))
        logger.info("Worker connected to supervisor at %s", self._socket_path)

        try:
            await self._serve(reader, writer)
        finally:
            self._running = False
            if self._scheduler:
                await self._scheduler.stop()
            if self._agent:
                await self._agent.shutdown()
            writer.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Main request loop — read messages, dispatch, respond."""
        while self._running:
            msg = await recv_message(reader)
            if msg is None:
                logger.info("Supervisor disconnected, shutting down")
                break

            if msg.type != "request":
                continue

            try:
                await self._handle_request(msg, writer)
            except Exception as exc:
                logger.error("Request %s failed: %s", msg.id, exc)
                await send_message(writer, Message.error(msg.id, traceback.format_exc()))

    async def _handle_request(self, msg: Message, writer: asyncio.StreamWriter) -> None:
        """Dispatch a single request."""
        method = msg.method

        if method == "health":
            await send_message(writer, Message.result(msg.id, {"ok": True}))

        elif method == "shutdown":
            logger.info("Shutdown requested, draining...")
            self._running = False
            await send_message(writer, Message.result(msg.id, {"ok": True}))

        elif method == "run":
            await self._handle_run(msg, writer)

        elif method == "feedback":
            await self._handle_feedback(msg, writer)

        else:
            await send_message(writer, Message.error(msg.id, f"Unknown method: {method}"))

    async def _handle_run(self, msg: Message, writer: asyncio.StreamWriter) -> None:
        """Execute agent.run() with progress streaming."""
        user_input = msg.params.get("user_input", "")
        history = msg.params.get("conversation_history", [])
        msg_id = msg.id

        async def _on_progress(phase: str, detail: str) -> None:
            await send_message(writer, Message.progress(msg_id, phase, detail))

        result = await self._agent.run(
            user_input,
            on_progress=_on_progress,
            conversation_history=history,
        )

        # Serialize AgentResult
        await send_message(writer, Message.result(msg_id, {
            "agent_result": result.model_dump(),
            "episode_id": self._agent.last_episode_id,
        }))

    async def _handle_feedback(self, msg: Message, writer: asyncio.StreamWriter) -> None:
        """Handle /good or /bad feedback."""
        score = msg.params.get("score", 9.0)
        reason = msg.params.get("reason")
        episode_id = msg.params.get("episode_id")

        if episode_id and self._agent.memory.available:
            await self._agent.memory.update_episode_score(episode_id, score, reason=reason)
            if reason and score < 5.0:
                lesson_rule = await self._agent.learn_from_feedback(reason)
                await send_message(writer, Message.result(msg.id, {
                    "ok": True, "lesson": lesson_rule,
                }))
                return

        await send_message(writer, Message.result(msg.id, {"ok": True}))


async def _main() -> None:
    load_dotenv()
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s [worker] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    # Parse --socket argument
    socket_path = "/tmp/nano-agent.sock"
    if "--socket" in sys.argv:
        idx = sys.argv.index("--socket")
        if idx + 1 < len(sys.argv):
            socket_path = sys.argv[idx + 1]

    worker = ExecutionWorker(socket_path, settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(_main())
