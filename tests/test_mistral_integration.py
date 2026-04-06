"""Integration tests for Mistral model preset.

Run: MODEL_NAME=MISTRAL python -m pytest tests/test_mistral_integration.py -v -s
Requires: Ollama running with mistral-small3.2 and ministral-3:8b models.
"""

import os
import pytest

# Force Mistral preset before any config import
os.environ["MODEL_NAME"] = "MISTRAL"

from agent.agent import Agent
from agent.config import get_settings


async def _run_agent(user_input: str, conversation_history=None):
    """Create a fresh agent, run a single query, and shut down."""
    settings = get_settings()
    agent = Agent(settings)
    await agent.initialize()
    try:
        return await agent.run(
            user_input, conversation_history=conversation_history
        )
    finally:
        await agent.shutdown()


# ── Test cases ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_greeting():
    """Simple triage path — direct answer, no planning."""
    result = await _run_agent("Salut, comment tu t'appelles ?")
    assert result.results, "Should produce at least one result"
    output = result.results[0].result.output
    assert len(output) > 5, f"Output too short: {output!r}"
    print(f"\n  [simple] Output: {output[:200]}")


@pytest.mark.asyncio
async def test_simple_factual():
    """Simple factual question."""
    result = await _run_agent("Quelle est la capitale de la France ?")
    assert result.results
    output = result.results[0].result.output.lower()
    assert "paris" in output, f"Expected 'paris' in: {output[:200]}"
    print(f"\n  [factual] Output: {output[:200]}")


@pytest.mark.asyncio
async def test_complex_planning():
    """Complex task — should trigger planning pipeline."""
    result = await _run_agent(
        "Compare les avantages et inconvénients de Python vs Rust "
        "pour le développement backend. Donne 3 points pour chaque."
    )
    assert result.results, "Should produce results"
    full_output = "\n".join(r.result.output for r in result.results)
    assert len(full_output) > 50, f"Output too short: {full_output[:100]}"
    print(f"\n  [complex] Steps: {len(result.results)}")
    for r in result.results:
        print(f"    Step {r.step.id} [{r.score.final_score:.1f}]: {r.result.output[:100]}...")


@pytest.mark.asyncio
async def test_conversation_history():
    """Multi-turn with conversation history."""
    history = [
        {"role": "user", "content": "Je travaille sur un projet Python."},
        {"role": "assistant", "content": "D'accord, je suis prêt à t'aider avec ton projet Python."},
    ]
    result = await _run_agent(
        "Quel framework web me conseilles-tu ?",
        conversation_history=history,
    )
    assert result.results
    output = result.results[0].result.output.lower()
    # Should mention at least one Python web framework
    frameworks = ["flask", "django", "fastapi", "starlette", "sanic", "bottle"]
    found = any(f in output for f in frameworks)
    assert found, f"Expected a framework name in: {output[:300]}"
    print(f"\n  [history] Output: {output[:200]}")


@pytest.mark.asyncio
async def test_lesson_detection():
    """Lesson extraction from user correction (verify no crash)."""
    history = [
        {"role": "user", "content": "Installe le package avec pip"},
        {"role": "assistant", "content": "D'accord, j'utilise pip install pour installer le package."},
    ]
    result = await _run_agent(
        "Non, utilise toujours uv au lieu de pip pour les dépendances Python",
        conversation_history=history,
    )
    assert result.results
    print(f"\n  [lesson] Output: {result.results[0].result.output[:200]}")


@pytest.mark.asyncio
async def test_critic_scores():
    """Verify critic produces valid scores on a complex task."""
    result = await _run_agent(
        "Écris une fonction Python qui calcule la factorielle d'un nombre"
    )
    assert result.results
    for r in result.results:
        assert 0 <= r.score.final_score <= 10, f"Score out of range: {r.score.final_score}"
    print(f"\n  [critic] Scores: {[r.score.final_score for r in result.results]}")
