"""Telegram bot interface — long polling, async job dispatch."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from agent.agent import Agent
from agent.config import Settings
from agent.core.job_manager import JobManager
from agent.personality.loader import Personality, load_personality
from agent.schemas import Job, JobStatus

logger = logging.getLogger(__name__)

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        cut = text.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len  # No good newline, hard cut
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# Phase labels for progress updates
_PHASE_LABELS = {
    "triage": "Analyse",
    "memory": "Contexte",
    "planning": "Planification",
    "executing": "Exécution",
    "critiquing": "Évaluation",
    "done": "Terminé",
    "failed": "Erreur",
}

# Minimum interval between progress edits (Telegram rate limit protection)
_PROGRESS_THROTTLE_S = 3.0


class TelegramBot:
    """Telegram bot that forwards messages to the Nano Agent via background jobs."""

    def __init__(self, agent: Agent, settings: Settings, job_manager: JobManager):
        if not HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot not installed. Run: pip install 'nano-agent[telegram]'"
            )

        self._agent = agent
        self._identity = getattr(agent, '_personality', load_personality()).identity
        self._settings = settings
        self._job_manager = job_manager
        self._max_len = settings.telegram.max_message_length
        self._allowed_users = set(settings.telegram.allowed_users)

        token = settings.telegram.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise RuntimeError(
                "No Telegram bot token. Set telegram.bot_token in config.yaml "
                "or TELEGRAM_BOT_TOKEN env var."
            )
        self._token = token
        self._app: Application | None = None

        # Track last progress edit time per job (for throttling)
        self._last_edit: dict[str, float] = {}

        # Conversation history per chat_id (last N turns)
        self._chat_history: dict[int, list[dict[str, str]]] = {}
        self._max_history_turns = 20

        # Register callbacks
        job_manager.set_progress_callback(self._on_job_progress)
        job_manager.set_completion_callback(self._on_job_complete)

    async def start(self) -> None:
        """Start the bot in long-polling mode (blocks forever)."""
        logger.info("Starting Telegram bot (long polling)...")
        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("good", self._handle_good))
        self._app.add_handler(CommandHandler("bad", self._handle_bad))
        self._app.add_handler(CommandHandler("stats", self._handle_stats))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("jobs", self._handle_status))
        self._app.add_handler(CommandHandler("cancel", self._handle_cancel))
        self._app.add_handler(CommandHandler("clean", self._handle_clean))
        self._app.add_handler(CommandHandler("evolve", self._handle_evolve))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        async with self._app:
            await self._app.initialize()
            await self._app.start()
            logger.info("Telegram bot is running. Send a message to interact.")
            await self._app.updater.start_polling(drop_pending_updates=True)

            # Block until stopped
            stop_event = asyncio.Event()

            def _signal_handler():
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig_name in ("SIGINT", "SIGTERM"):
                try:
                    import signal
                    loop.add_signal_handler(
                        getattr(signal, sig_name), _signal_handler
                    )
                except (NotImplementedError, AttributeError):
                    pass

            await stop_event.wait()
            logger.info("Shutting down Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()

    def _is_allowed(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    # -- Command handlers ------------------------------------------------------

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Nano Agent ready. Send me a message and I'll process it "
            "in background.\n\n/help for more info."
        )

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Send me any task or question.\n\n"
            "Simple questions get a fast answer.\n"
            "Complex tasks run in background — I stay available.\n\n"
            "Commands:\n"
            "/status — see running jobs\n"
            "/cancel [job_id] — cancel a running job\n"
            "/clean — clear conversation history\n"
            "/good — validate the last result\n"
            "/bad [reason] — reject the last result\n"
            "/stats — show performance metrics\n"
            "/evolve — evolve workspace from lessons\n\n"
            "Examples:\n"
            "• What is the capital of France?\n"
            "• Write a Python function to sort a list\n"
            "• Search for Python files in the project"
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Acknowledge immediately, dispatch job in background."""
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name or str(user_id)
        text = update.message.text

        if not self._is_allowed(user_id):
            await update.message.reply_text("Access denied.")
            logger.warning("Denied user %d (%s)", user_id, user_name)
            return

        logger.info("Message from %s (%d): %s", user_name, user_id, text[:80])

        chat_id = update.effective_chat.id

        # Get conversation history for this chat
        history = self._chat_history.get(chat_id, [])

        # Add user message to history
        if chat_id not in self._chat_history:
            self._chat_history[chat_id] = []
        self._chat_history[chat_id].append({"role": "user", "content": text})

        # Immediate acknowledgment
        ack = await update.message.reply_text("Je travaille dessus...")

        # Submit background job with conversation history
        job = await self._job_manager.submit(
            user_input=text,
            chat_id=chat_id,
            message_id=ack.message_id,
            conversation_history=list(history),  # Pass copy of history before this message
        )
        logger.info("Job %s submitted for %s", job.id, user_name)

    async def _handle_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show active and recent jobs for this chat."""
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id
        active = self._job_manager.get_active_jobs(chat_id)
        recent = self._job_manager.get_recent_jobs(chat_id, limit=5)

        lines: list[str] = []
        if active:
            lines.append("Jobs en cours:")
            for j in active:
                phase = _PHASE_LABELS.get(j.current_phase, j.current_phase)
                lines.append(f"  • {j.id[:8]} — {phase} — {j.progress}")
                lines.append(f"    \"{j.user_input[:60]}\"")
        else:
            lines.append("Aucun job en cours.")

        done = [j for j in recent if j.status in (JobStatus.done, JobStatus.failed)]
        if done:
            lines.append("\nRécents:")
            for j in done[:3]:
                icon = "✓" if j.status == JobStatus.done else "✗"
                lines.append(f"  {icon} {j.id[:8]} — \"{j.user_input[:60]}\"")

        await update.message.reply_text("\n".join(lines))

    async def _handle_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Cancel a running job by ID prefix."""
        if not self._is_allowed(update.effective_user.id):
            return

        if not context.args:
            # Cancel most recent active job for this chat
            active = self._job_manager.get_active_jobs(update.effective_chat.id)
            if not active:
                await update.message.reply_text("Aucun job à annuler.")
                return
            job_id = active[0].id
        else:
            # Find job matching prefix
            prefix = context.args[0]
            job_id = None
            for j in self._job_manager.get_active_jobs():
                if j.id.startswith(prefix):
                    job_id = j.id
                    break
            if not job_id:
                await update.message.reply_text(f"Job '{prefix}' not found.")
                return

        cancelled = await self._job_manager.cancel_job(job_id)
        if cancelled:
            await update.message.reply_text(f"Job {job_id[:8]} annulé.")
        else:
            await update.message.reply_text(f"Job {job_id[:8]} déjà terminé.")

    async def _handle_clean(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Clear conversation history for this chat."""
        if not self._is_allowed(update.effective_user.id):
            return
        chat_id = update.effective_chat.id
        count = len(self._chat_history.get(chat_id, []))
        self._chat_history.pop(chat_id, None)
        await update.message.reply_text(
            f"Conversation history cleared ({count} messages removed)."
        )
        logger.info("Chat history cleared for chat %d (%d messages)", chat_id, count)

    async def _handle_good(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        episode_id = self._agent.last_episode_id
        if not episode_id:
            await update.message.reply_text("No recent result to rate.")
            return
        if self._agent.memory.available:
            await self._agent.memory.update_episode_score(episode_id, 9.0)
        await update.message.reply_text("Noted. I'll remember what worked.")
        logger.info("Feedback: /good on episode %s", episode_id)

    async def _handle_bad(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        episode_id = self._agent.last_episode_id
        if not episode_id:
            await update.message.reply_text("No recent result to rate.")
            return
        reason = " ".join(context.args) if context.args else None
        if self._agent.memory.available:
            await self._agent.memory.update_episode_score(
                episode_id, 2.0, reason=reason
            )
        reply = "Got it. I'll do better next time."
        if reason:
            lesson_rule = await self._agent.learn_from_feedback(reason)
            if lesson_rule:
                reply += f"\nLesson learned: {lesson_rule}"
        await update.message.reply_text(reply)
        logger.info("Feedback: /bad on episode %s (reason: %s)", episode_id, reason)

    async def _handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        from agent.infra.metrics import MetricsCollector
        mc = MetricsCollector(self._settings.logging.trace_file)
        summary = mc.summary(last_n=50)
        if summary.get("total_episodes", 0) == 0:
            await update.message.reply_text("No data yet.")
            return
        lines = [f"Last {summary['total_episodes']} episodes:"]
        lines.append(f"  Avg score: {summary.get('avg_score', 0):.1f}")
        lines.append(f"  Avg latency: {summary.get('avg_latency_s', 0):.1f}s")
        lines.append(f"  Retry rate: {summary.get('retry_rate', 0):.0%}")
        by_type = summary.get("by_task_type", {})
        if by_type:
            lines.append("  By type:")
            for k, v in by_type.items():
                lines.append(f"    {k}: {v:.1f}")
        await update.message.reply_text("\n".join(lines))

    async def _handle_evolve(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        await update.message.reply_text("Évolution du workspace en cours...")
        try:
            modified = await self._agent._evolve_workspace()
            if modified:
                await update.message.reply_text(f"Workspace mis à jour : {', '.join(modified)}")
            else:
                await update.message.reply_text("Pas de changement — pas assez de lessons.")
        except Exception as exc:
            await update.message.reply_text(f"Erreur : {exc}")

    # -- Job callbacks ---------------------------------------------------------

    async def _on_job_progress(self, job: Job) -> None:
        """Edit the ack message with progress (throttled)."""
        if not self._app or not job.message_id:
            return

        # Throttle edits to avoid Telegram rate limits
        now = time.monotonic()
        last = self._last_edit.get(job.id, 0)
        if now - last < _PROGRESS_THROTTLE_S:
            return
        self._last_edit[job.id] = now

        phase = _PHASE_LABELS.get(job.current_phase, job.current_phase)
        text = f"En cours... {phase}"
        if job.progress:
            text += f" — {job.progress}"

        try:
            await self._app.bot.edit_message_text(
                chat_id=job.chat_id,
                message_id=job.message_id,
                text=text,
            )
        except Exception:
            pass  # Edit can fail if message is identical or too old

    async def _on_job_complete(self, job: Job) -> None:
        """Send final result as a new message (triggers notification)."""
        if not self._app:
            return

        # Clean up throttle tracking
        self._last_edit.pop(job.id, None)

        if job.status == JobStatus.done and job.result:
            # Edit ack to show completion
            if job.message_id:
                try:
                    await self._app.bot.edit_message_text(
                        chat_id=job.chat_id,
                        message_id=job.message_id,
                        text="Terminé",
                    )
                except Exception:
                    pass

            # Send result as new message (triggers push notification)
            response = self._format_result(job.result)
            for chunk in _split_message(response, self._max_len):
                await self._app.bot.send_message(chat_id=job.chat_id, text=chunk)

            # Add assistant response to conversation history
            assistant_output = "\n".join(
                r.result.output[:300] for r in job.result.results
            )
            if job.chat_id in self._chat_history:
                self._chat_history[job.chat_id].append(
                    {"role": "assistant", "content": assistant_output}
                )
                # Trim history to max turns
                self._chat_history[job.chat_id] = (
                    self._chat_history[job.chat_id][-self._max_history_turns:]
                )

            logger.info(
                "Job %s result sent: %d steps, scores=%s",
                job.id,
                len(job.result.results),
                [r.score.final_score for r in job.result.results],
            )

        elif job.status == JobStatus.failed:
            error_msg = f"Erreur: {job.error or 'Unknown error'}"
            if job.message_id:
                try:
                    await self._app.bot.edit_message_text(
                        chat_id=job.chat_id,
                        message_id=job.message_id,
                        text=error_msg,
                    )
                except Exception:
                    await self._app.bot.send_message(
                        chat_id=job.chat_id, text=error_msg
                    )
            else:
                await self._app.bot.send_message(
                    chat_id=job.chat_id, text=error_msg
                )

    # -- Formatting ------------------------------------------------------------

    def _format_result(self, result) -> str:
        """Format an AgentResult into a readable Telegram message."""
        lines: list[str] = []

        for sr in result.results:
            output = sr.result.output.strip()
            if not output:
                continue
            if len(output) > 1500:
                output = output[:1500] + "\n... (truncated)"
            lines.append(output)

        return "\n\n".join(lines).strip()
