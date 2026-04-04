"""CLI entry point.

Usage:
    python -m agent "your prompt here"     # One-shot mode
    python -m agent --daemon               # Telegram bot daemon (24/7)
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.agent import Agent
from agent.config import get_settings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m agent 'your prompt'   # One-shot")
        print("  python -m agent --daemon         # Telegram bot (24/7)")
        sys.exit(1)

    if sys.argv[1] == "--daemon":
        _run_daemon()
    else:
        _run_oneshot()


def _run_daemon() -> None:
    """Start the Telegram bot in daemon mode."""
    from agent.daemon import run_daemon
    asyncio.run(run_daemon())


def _run_oneshot() -> None:
    """Run a single prompt and exit."""
    user_input = " ".join(sys.argv[1:])

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    async def _run():
        agent = Agent(settings)
        await agent.initialize()
        try:
            result = await agent.run(user_input)
        finally:
            await agent.shutdown()
        return result

    result = asyncio.run(_run())

    print("\n" + "=" * 60)
    print(f"Goal: {result.goal}")
    print("=" * 60)
    for sr in result.results:
        score_str = f"[{sr.score.final_score:.1f}/10]"
        print(f"\nStep {sr.step.id} {score_str} ({sr.step.tool}):")
        print(sr.result.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
