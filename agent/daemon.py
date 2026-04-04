"""Daemon mode — runs the Telegram bot as a long-lived service."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from agent.agent import Agent
from agent.config import Settings, get_settings
from agent.interfaces.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env file from project root into os.environ."""
    import os
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


async def run_daemon(settings: Settings | None = None) -> None:
    """Start the agent in daemon mode with Telegram bot."""
    _load_dotenv()
    settings = settings or get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Nano Agent Daemon starting ===")

    agent = Agent(settings)
    await agent.initialize()

    try:
        bot = TelegramBot(agent, settings)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await agent.shutdown()
        logger.info("=== Nano Agent Daemon stopped ===")
