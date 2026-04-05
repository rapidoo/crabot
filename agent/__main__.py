"""CLI entry point.

Usage:
    python -m agent                        # Interactive REPL
    python -m agent "your prompt here"     # One-shot mode
    python -m agent --daemon               # Telegram bot daemon (24/7)
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.agent import Agent
from agent.config import get_settings
from agent.env import load_dotenv
from agent.personality.loader import load_personality


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        _run_repl()
    elif sys.argv[1] == "--daemon":
        _run_daemon()
    elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
        _print_help()
    else:
        _run_oneshot()


def _print_help() -> None:
    identity = load_personality().identity
    print(f"{identity.emoji} {identity.name} — Agent IA local")
    print()
    print("Usage:")
    print("  python -m agent                  Interactive (REPL)")
    print("  python -m agent 'prompt'         One-shot")
    print("  python -m agent --daemon         Telegram bot (24/7)")
    print()
    print("REPL commands:")
    print("  /tools       List available tools")
    print("  /stats       Show performance metrics")
    print("  /goals       List active goals")
    print("  /goal <desc> Create a new goal")
    print("  /quit        Exit")


def _run_daemon() -> None:
    from agent.daemon import run_daemon
    asyncio.run(run_daemon())


def _run_oneshot() -> None:
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
    _print_result(result)


def _run_repl() -> None:
    """Interactive REPL — chat with the agent in the terminal."""
    settings = get_settings()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    identity = load_personality().identity
    print(f"\n{identity.emoji} {identity.name} — Agent IA local")
    print(f"Tape ta question ou /help pour les commandes. /quit pour sortir.\n")

    asyncio.run(_repl_loop(settings, identity))


async def _repl_loop(settings, identity) -> None:
    agent = Agent(settings)
    await agent.initialize()

    try:
        while True:
            try:
                user_input = input(f"{identity.emoji} > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                handled = await _handle_command(user_input, agent)
                if handled == "quit":
                    break
                continue

            # Run the agent
            try:
                result = await agent.run(user_input)
                _print_result(result, compact=True)
            except Exception as exc:
                print(f"\n  Error: {exc}\n")

    finally:
        await agent.shutdown()


async def _handle_command(cmd: str, agent: Agent) -> str | None:
    """Handle REPL slash commands. Returns 'quit' to exit."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        print("Bye.")
        return "quit"

    elif command == "/help":
        print("  /tools       List available tools")
        print("  /stats       Show performance metrics")
        print("  /goals       List active goals")
        print("  /goal <desc> Create a new goal")
        print("  /good        Validate last result")
        print("  /bad [why]   Reject last result")
        print("  /quit        Exit")

    elif command == "/tools":
        from agent.tools.registry import list_tools_with_descriptions
        tools = list_tools_with_descriptions()
        if tools:
            for t in tools:
                print(f"  {t['name']:15s} {t['description']}")
        else:
            print("  No tools registered.")

    elif command == "/stats":
        from agent.infra.metrics import MetricsCollector
        mc = MetricsCollector(agent._settings.logging.trace_file)
        summary = mc.summary(last_n=50)
        if summary.get("total_episodes", 0) == 0:
            print("  No data yet.")
        else:
            print(f"  Episodes: {summary['total_episodes']}")
            print(f"  Avg score: {summary.get('avg_score', 0):.1f}")
            print(f"  Avg latency: {summary.get('avg_latency_s', 0):.1f}s")
            print(f"  Retry rate: {summary.get('retry_rate', 0):.0%}")

    elif command == "/goals":
        if not agent._memory.available:
            print("  Neo4j not connected.")
        else:
            goals = await agent._memory.get_active_goals()
            if not goals:
                print("  No active goals.")
            else:
                for g in goals:
                    print(f"  [{g.get('priority', '?')}] {g.get('description', '?')}")

    elif command == "/goal":
        if not args:
            print("  Usage: /goal <description>")
        elif not agent._memory.available:
            print("  Neo4j not connected.")
        else:
            import hashlib
            goal_id = hashlib.md5(args.encode()).hexdigest()[:8]
            await agent._memory.persist_goal(goal_id, args)
            print(f"  Goal created: {args}")

    elif command == "/good":
        ep = agent.last_episode_id
        if not ep:
            print("  No recent result to rate.")
        elif agent._memory.available:
            await agent._memory.update_episode_score(ep, 9.0)
            print("  Noted — positive feedback saved.")
        else:
            print("  Neo4j not connected.")

    elif command == "/bad":
        ep = agent.last_episode_id
        if not ep:
            print("  No recent result to rate.")
        elif agent._memory.available:
            await agent._memory.update_episode_score(ep, 2.0, reason=args or None)
            print("  Noted — negative feedback saved.")
        else:
            print("  Neo4j not connected.")

    else:
        print(f"  Unknown command: {command}. Type /help")

    return None


def _print_result(result, compact: bool = False) -> None:
    """Print an AgentResult to the terminal."""
    if compact:
        print()
        for sr in result.results:
            score = sr.score.final_score
            icon = "✓" if score >= 6.5 else "✗"
            if len(result.results) > 1:
                print(f"  {icon} Step {sr.step.id} [{score:.0f}/10]:")
            print(f"  {sr.result.output}")
            print()
    else:
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
