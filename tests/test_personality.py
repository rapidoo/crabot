"""Tests for personality loader."""

from pathlib import Path

from agent.personality.loader import load_personality, _parse_identity, Identity


class TestLoadPersonality:
    def test_load_from_workspace(self):
        p = load_personality()
        assert len(p.soul) > 0
        assert "Crabot" in p.identity.name
        assert p.identity.emoji == "🦀"
        assert len(p.agents) > 0
        assert "Frédéric" in p.user

    def test_load_missing_workspace(self, tmp_path):
        p = load_personality(tmp_path / "nonexistent")
        assert p.soul == ""
        assert p.identity.name == "Agent"

    def test_parse_identity(self, tmp_path):
        f = tmp_path / "IDENTITY.md"
        f.write_text("- Name: TestBot\n- Emoji: 🔥\n- Creature: phoenix\n- Vibe: fiery\n- Theme: red")
        identity = _parse_identity(f)
        assert identity.name == "TestBot"
        assert identity.emoji == "🔥"
        assert identity.creature == "phoenix"

    def test_heartbeat_loaded(self):
        p = load_personality()
        assert len(p.heartbeat) > 0


class TestIdentity:
    def test_defaults(self):
        i = Identity()
        assert i.name == "Agent"
        assert i.emoji == "🤖"
