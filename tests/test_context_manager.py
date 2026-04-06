"""Tests for the ContextManager — multi-turn history + compression."""

import pytest

from agent.core.context_manager import ContextManager, Turn


class TestContextManager:
    def test_add_turn_stores_history(self):
        cm = ContextManager(max_tokens=100_000)
        cm.add_turn("hello", "world")
        assert len(cm.turns) == 1
        assert cm.turns[0].user == "hello"
        assert cm.turns[0].assistant == "world"

    def test_get_history_messages_empty(self):
        cm = ContextManager()
        assert cm.get_history_messages() == []

    def test_get_history_messages_with_turns(self):
        cm = ContextManager(max_tokens=100_000)
        cm.add_turn("q1", "a1")
        cm.add_turn("q2", "a2")
        msgs = cm.get_history_messages()
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "q1"}
        assert msgs[1] == {"role": "assistant", "content": "a1"}
        assert msgs[2] == {"role": "user", "content": "q2"}
        assert msgs[3] == {"role": "assistant", "content": "a2"}

    def test_compression_triggers_when_over_budget(self):
        # Small budget to trigger compression easily
        cm = ContextManager(
            max_tokens=50,  # ~200 chars budget, threshold 50% = ~100 chars
            compression_threshold=0.50,
            protect_last_n=2,
        )
        # Each turn is ~20 chars, so 6 turns = ~120 chars > 100 budget
        for i in range(6):
            cm.add_turn(f"question_{i}", f"answer_{i}")

        # Should have compressed older turns
        assert len(cm.turns) <= 4  # protect_last_n=2, but compression is lazy
        assert cm.summary_block != ""

    def test_compression_keeps_last_n(self):
        cm = ContextManager(
            max_tokens=20,
            compression_threshold=0.50,
            protect_last_n=2,
        )
        cm.add_turn("q1", "a1")
        cm.add_turn("q2", "a2")
        cm.add_turn("q3", "a3")
        cm.add_turn("q4", "a4")

        # Last 2 should always be preserved
        assert cm.turns[-1].user == "q4"
        assert cm.turns[-2].user == "q3" or len(cm.turns) == 2

    def test_summary_block_in_messages(self):
        cm = ContextManager(
            max_tokens=20,
            compression_threshold=0.50,
            protect_last_n=1,
        )
        cm.add_turn("q1", "a1")
        cm.add_turn("q2", "a2")
        cm.add_turn("q3", "a3")

        msgs = cm.get_history_messages()
        # First message should be the summary
        if cm.summary_block:
            assert msgs[0]["role"] == "system"
            assert "Summary" in msgs[0]["content"]

    def test_estimated_tokens(self):
        cm = ContextManager()
        cm.add_turn("hello", "world")
        # "hello" + "world" = 10 chars, ~2-3 tokens
        assert cm.estimated_tokens() > 0

    def test_clear(self):
        cm = ContextManager(max_tokens=100_000)
        cm.add_turn("q", "a")
        cm.summary_block = "old stuff"
        cm.clear()
        assert len(cm.turns) == 0
        assert cm.summary_block == ""

    def test_no_compression_when_under_budget(self):
        cm = ContextManager(
            max_tokens=100_000,
            compression_threshold=0.50,
            protect_last_n=2,
        )
        cm.add_turn("q1", "a1")
        cm.add_turn("q2", "a2")
        assert len(cm.turns) == 2
        assert cm.summary_block == ""

    def test_turn_char_count(self):
        t = Turn(user="hello", assistant="world!")
        assert t.char_count() == 11
