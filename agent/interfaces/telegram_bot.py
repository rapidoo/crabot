"""Telegram bot interface — long polling, wraps the Agent pipeline."""

from __future__ import annotations

import asyncio
import logging
import os

from agent.agent import Agent
from agent.config import Settings

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


class TelegramBot:
    """Telegram bot that forwards messages to the Nano Agent."""

    def __init__(self, agent: Agent, settings: Settings):
        if not HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot not installed. Run: pip install 'nano-agent[telegram]'"
            )

        self._agent = agent
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.daemon.max_concurrent)
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
            return True  # No filter = everyone allowed
        return user_id in self._allowed_users

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Nano Agent ready. Send me a message and I'll process it through "
            "the Plan → Execute → Critique pipeline.\n\n"
            "/help for more info."
        )

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Send me any task or question.\n\n"
            "Simple questions get a fast answer (~1s).\n"
            "Complex tasks go through planning, execution, and critique.\n\n"
            "Examples:\n"
            "• What is the capital of France?\n"
            "• Write a Python function to sort a list\n"
            "• Search for Python files in the project"
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name or str(user_id)
        text = update.message.text

        if not self._is_allowed(user_id):
            await update.message.reply_text("Access denied.")
            logger.warning("Denied user %d (%s)", user_id, user_name)
            return

        logger.info("Message from %s (%d): %s", user_name, user_id, text[:80])

        # Send typing indicator
        await update.message.chat.send_action("typing")

        try:
            async with self._semaphore:
                result = await asyncio.wait_for(
                    self._agent.run(text),
                    timeout=self._settings.daemon.request_timeout,
                )

            # Format response
            response = self._format_result(result)

            # Split and send
            for chunk in _split_message(response, self._max_len):
                await update.message.reply_text(chunk)

            logger.info(
                "Reply to %s: %d steps, scores=%s",
                user_name,
                len(result.results),
                [r.score.final_score for r in result.results],
            )

        except asyncio.TimeoutError:
            await update.message.reply_text(
                "Timeout — the task took too long. Try a simpler request."
            )
            logger.warning("Timeout for user %s on: %s", user_name, text[:80])

        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")
            logger.error("Error processing message from %s: %s", user_name, exc)

    def _format_result(self, result) -> str:
        """Format an AgentResult into a readable Telegram message."""
        lines: list[str] = []
        lines.append(f"Goal: {result.goal}\n")

        for sr in result.results:
            score = sr.score.final_score
            icon = "✓" if score >= 6.5 else "✗"
            lines.append(f"{icon} Step {sr.step.id} [{score:.0f}/10] ({sr.step.tool}):")
            # Truncate long outputs
            output = sr.result.output
            if len(output) > 1500:
                output = output[:1500] + "\n... (truncated)"
            lines.append(output)
            lines.append("")

        return "\n".join(lines).strip()
