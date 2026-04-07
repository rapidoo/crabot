"""Supervisor — manages worker lifecycle and routes IPC messages.

The supervisor owns user-facing interfaces (Telegram, CLI) and delegates
execution to a restartable worker subprocess. This enables the agent to
modify its own code and reload by restarting the worker.

Started via:
    python -m agent --supervisor
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agent.config import Settings, get_settings
from agent.ipc import Message, send_message, recv_message, recv_message_timeout
from agent.schemas import AgentResult

logger = logging.getLogger(__name__)

# How long to wait for worker to drain jobs on shutdown
_DRAIN_TIMEOUT = 30.0
# How long between health check pings
_HEALTH_INTERVAL = 30.0
# Max consecutive health failures before restart
_MAX_HEALTH_FAILURES = 3


class WorkerProxy:
    """Manages a single execution worker subprocess."""

    def __init__(self, socket_path: str, settings: Settings):
        self._socket_path = socket_path
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._server: asyncio.Server | None = None
        self._ready = asyncio.Event()
        self._alive = False
        self._restart_count = 0
        self._max_restarts = 5

    async def start(self) -> None:
        """Start the Unix socket server and spawn the worker."""
        # Clean stale socket
        sock_path = Path(self._socket_path)
        if sock_path.exists():
            sock_path.unlink()

        # Start socket server (accepts one connection from worker)
        self._ready.clear()
        self._server = await asyncio.start_unix_server(
            self._on_worker_connect, path=self._socket_path
        )
        logger.info("Supervisor: listening on %s", self._socket_path)

        # Spawn worker subprocess
        self._spawn_worker()

        # Wait for worker to connect and send ready
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Worker failed to connect within 30s")
            raise RuntimeError("Worker startup timeout")

        logger.info("Supervisor: worker is ready (PID %d)", self._process.pid)

    def _spawn_worker(self) -> None:
        """Launch the worker subprocess."""
        cmd = [
            sys.executable, "-m", "agent.worker",
            "--socket", self._socket_path,
        ]
        # Forward MODEL_NAME if set
        env = os.environ.copy()
        self._process = subprocess.Popen(
            cmd, env=env, stdout=sys.stdout, stderr=sys.stderr,
        )
        logger.info("Supervisor: spawned worker PID %d", self._process.pid)

    async def _on_worker_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Called when the worker connects to our socket."""
        self._reader = reader
        self._writer = writer
        self._alive = True

        # Wait for ready message
        msg = await recv_message_timeout(reader, timeout=10.0)
        if msg and msg.method == "ready":
            self._ready.set()
        else:
            logger.warning("Worker connected but no ready message: %s", msg)
            self._ready.set()  # proceed anyway

    async def send_request(
        self,
        method: str,
        timeout: float = 300.0,
        **params,
    ) -> Message | None:
        """Send a request to the worker and wait for the result.

        Yields progress messages via the optional on_progress callback in params.
        Returns the final result/error message, or None on timeout.
        """
        if not self._writer or not self._reader:
            return None

        msg = Message.request(method, **params)
        await send_message(self._writer, msg)

        on_progress = params.pop("on_progress", None)

        # Collect response(s) — progress messages followed by result/error
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            resp = await recv_message_timeout(self._reader, timeout=remaining)
            if resp is None:
                return None
            if resp.id != msg.id:
                continue  # skip messages from other requests
            if resp.type == "progress":
                if on_progress:
                    try:
                        phase = resp.data.get("phase", "")
                        detail = resp.data.get("detail", "")
                        ret = on_progress(phase, detail)
                        if ret is not None:
                            await ret
                    except Exception:
                        pass
                continue
            return resp  # result or error
        return None

    async def health_check(self) -> bool:
        """Ping the worker and check if it responds."""
        if not self._alive:
            return False
        resp = await self.send_request("health", timeout=5.0)
        return resp is not None and resp.type == "result"

    async def restart(self) -> bool:
        """Gracefully restart the worker (drain + respawn)."""
        if self._restart_count >= self._max_restarts:
            logger.error("Max restart attempts reached (%d)", self._max_restarts)
            return False

        self._restart_count += 1
        logger.info("Supervisor: restarting worker (attempt %d)", self._restart_count)

        # Try graceful shutdown first
        await self._shutdown_worker()

        # Close old connection
        self._alive = False
        if self._writer:
            self._writer.close()
        self._reader = None
        self._writer = None

        # Respawn
        self._ready.clear()
        self._spawn_worker()
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Worker restart failed: startup timeout")
            return False

        logger.info("Supervisor: worker restarted (PID %d)", self._process.pid)
        return True

    async def _shutdown_worker(self) -> None:
        """Send shutdown to worker and wait for exit."""
        if self._writer and self._alive:
            try:
                resp = await self.send_request("shutdown", timeout=_DRAIN_TIMEOUT)
                if resp:
                    logger.info("Worker acknowledged shutdown")
            except Exception:
                pass

        # Wait for process to exit
        if self._process and self._process.poll() is None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Worker didn't exit, sending SIGTERM")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Worker didn't terminate, sending SIGKILL")
                    self._process.kill()

    async def stop(self) -> None:
        """Fully stop the worker and clean up."""
        await self._shutdown_worker()
        self._alive = False
        if self._writer:
            self._writer.close()
        if self._server:
            self._server.close()
        # Clean socket file
        sock = Path(self._socket_path)
        if sock.exists():
            sock.unlink()


class Supervisor:
    """Top-level supervisor that routes interfaces to the worker."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        socket_path = getattr(
            self._settings, "supervisor", None
        )
        if socket_path and hasattr(socket_path, "socket_path"):
            self._socket_path = socket_path.socket_path
        else:
            self._socket_path = "/tmp/nano-agent.sock"
        self._worker = WorkerProxy(self._socket_path, self._settings)
        self._stopping = False

    async def start(self) -> None:
        """Start the supervisor, worker, and interfaces."""
        await self._worker.start()

        # Start health check loop
        health_task = asyncio.create_task(self._health_loop())

        try:
            # Start Telegram bot if configured
            await self._run_telegram()
        except KeyboardInterrupt:
            logger.info("Supervisor interrupted")
        finally:
            self._stopping = True
            health_task.cancel()
            await self._worker.stop()
            logger.info("=== Supervisor stopped ===")

    async def _health_loop(self) -> None:
        """Periodically check worker health, restart if unhealthy."""
        failures = 0
        while not self._stopping:
            await asyncio.sleep(_HEALTH_INTERVAL)
            if self._stopping:
                break
            ok = await self._worker.health_check()
            if ok:
                failures = 0
            else:
                failures += 1
                logger.warning("Health check failed (%d/%d)", failures, _MAX_HEALTH_FAILURES)
                if failures >= _MAX_HEALTH_FAILURES:
                    logger.error("Worker unresponsive, restarting...")
                    await self._worker.restart()
                    failures = 0

    async def run_agent(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
        on_progress=None,
    ) -> AgentResult | None:
        """Route an agent execution request to the worker."""
        resp = await self._worker.send_request(
            "run",
            timeout=self._settings.daemon.request_timeout,
            user_input=user_input,
            conversation_history=conversation_history or [],
            on_progress=on_progress,
        )
        if resp is None:
            logger.error("Worker timeout for: %s", user_input[:80])
            return None
        if resp.type == "error":
            logger.error("Worker error: %s", resp.error[:200])
            return None
        # Deserialize AgentResult
        agent_data = resp.data.get("agent_result")
        if agent_data:
            return AgentResult.model_validate(agent_data)
        return None

    async def send_feedback(self, episode_id: str, score: float, reason: str | None = None) -> str | None:
        """Route feedback to the worker."""
        resp = await self._worker.send_request(
            "feedback", timeout=10.0,
            episode_id=episode_id, score=score, reason=reason,
        )
        if resp and resp.type == "result":
            return resp.data.get("lesson")
        return None

    async def request_evolution(self, mutations: list[dict]) -> bool:
        """Trigger code evolution: spawn evolution worker, validate, restart."""
        logger.info("Evolution requested: %d mutations", len(mutations))

        # Spawn evolution worker as subprocess
        import json
        cmd = [
            sys.executable, "-m", "agent.worker_evolution",
            "--mutations", json.dumps(mutations),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)

        if proc.returncode != 0:
            logger.error("Evolution failed:\n%s", stderr.decode()[:500])
            return False

        # Parse result
        try:
            result = json.loads(stdout.decode())
        except json.JSONDecodeError:
            logger.error("Evolution: invalid JSON output")
            return False

        if not result.get("ok"):
            logger.error("Evolution validation failed: %s", result.get("error", "unknown"))
            return False

        # Restart worker with fresh code
        logger.info("Evolution validated, restarting worker...")
        success = await self._worker.restart()
        if success:
            logger.info("Evolution complete — worker reloaded with new code")
        return success

    async def _run_telegram(self) -> None:
        """Start the Telegram bot, routing messages through the worker."""
        token = self._settings.telegram.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.warning("No Telegram token — supervisor running without bot")
            # Keep alive for health checks
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    pass
            await stop.wait()
            return

        # Import and start the Telegram bot, using the supervisor as agent proxy
        from agent.interfaces.telegram_bot import TelegramBot
        from agent.core.job_manager import JobManager

        # Create a lightweight agent proxy for the bot
        proxy = _AgentProxy(self)
        job_manager = JobManager(
            agent=proxy,
            memory=_MemoryStub(),
            max_concurrent=self._settings.daemon.max_concurrent,
        )
        bot = TelegramBot(proxy, self._settings, job_manager=job_manager)
        await bot.start()


class _AgentProxy:
    """Lightweight proxy that makes the Supervisor look like an Agent to TelegramBot."""

    def __init__(self, supervisor: Supervisor):
        self._supervisor = supervisor
        self._last_episode_id: str | None = None

    @property
    def last_episode_id(self) -> str | None:
        return self._last_episode_id

    @property
    def memory(self):
        return _MemoryStub()

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run(
        self,
        user_input: str,
        on_progress=None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        result = await self._supervisor.run_agent(
            user_input, conversation_history, on_progress,
        )
        if result is None:
            from agent.schemas import Step, StepResult, CriticScore, ScoredResult
            result = AgentResult(
                goal=user_input[:100],
                results=[ScoredResult(
                    step=Step(id=0, tool="none", input=user_input, expected_output=""),
                    result=StepResult(step_id=0, output="Worker unavailable. Please retry."),
                    score=CriticScore(scores={}, final_score=0, retry=False),
                )],
            )
        return result

    async def learn_from_feedback(self, reason: str) -> str | None:
        return await self._supervisor.send_feedback(
            self._last_episode_id or "", 2.0, reason,
        )


class _MemoryStub:
    """Stub for memory client — the real one is in the worker."""
    available = False

    async def update_episode_score(self, *a, **kw):
        pass
