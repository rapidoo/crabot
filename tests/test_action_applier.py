"""Tests for ActionApplier — apply/rollback reflection actions."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.config import Settings
from agent.intelligence.action_applier import ActionApplier


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.evolution.enabled = True
    s.evolution.overrides_file = str(tmp_path / "overrides.yaml")
    s.evolution.mutations_log = str(tmp_path / "mutations.jsonl")
    return s


@pytest.fixture
def applier(settings):
    return ActionApplier(settings)


class TestActionApplier:
    @pytest.mark.asyncio
    async def test_disabled_skips(self, tmp_path):
        s = Settings()
        s.evolution.enabled = False
        s.evolution.overrides_file = str(tmp_path / "overrides.yaml")
        s.evolution.mutations_log = str(tmp_path / "mutations.jsonl")
        applier = ActionApplier(s)
        result = await applier.apply([{"type": "adjust_threshold", "new_value": 5.0}])
        assert result == []

    @pytest.mark.asyncio
    async def test_adjust_threshold(self, applier, settings):
        actions = [{"type": "adjust_threshold", "new_value": 7.5, "reason": "test"}]
        mutations = await applier.apply(actions)
        assert len(mutations) == 1
        assert mutations[0].action_type == "adjust_threshold"
        assert settings.thresholds.min_score == 7.5

    @pytest.mark.asyncio
    async def test_threshold_clamped(self, applier, settings):
        await applier.apply([{"type": "adjust_threshold", "new_value": 15.0}])
        assert settings.thresholds.min_score == 9.0

        await applier.apply([{"type": "adjust_threshold", "new_value": 1.0}])
        assert settings.thresholds.min_score == 4.0

    @pytest.mark.asyncio
    async def test_escalate_model(self, applier, settings):
        actions = [{"type": "escalate_model", "task_type": "triage", "to": "gemma4:26b"}]
        mutations = await applier.apply(actions)
        assert len(mutations) == 1
        assert settings.models.triage == "gemma4:26b"

    @pytest.mark.asyncio
    async def test_protected_role_blocked(self, applier, settings):
        actions = [{"type": "escalate_model", "task_type": "critic", "to": "gemma4:e4b"}]
        mutations = await applier.apply(actions)
        assert len(mutations) == 0
        assert settings.models.critic == "gemma4:31b"  # Unchanged

    @pytest.mark.asyncio
    async def test_disable_tool(self, applier):
        actions = [{"type": "disable_tool", "tool": "search", "reason": "broken"}]
        mutations = await applier.apply(actions)
        assert len(mutations) == 1
        assert "search" in applier.disabled_tools

    @pytest.mark.asyncio
    async def test_create_skill_with_memory(self, settings):
        memory = AsyncMock()
        memory.available = True
        memory.persist_skill = AsyncMock()
        applier = ActionApplier(settings, memory=memory)
        actions = [{"type": "create_skill", "name": "my_skill", "tool_chain": ["search", "code"]}]
        mutations = await applier.apply(actions)
        assert len(mutations) == 1
        memory.persist_skill.assert_awaited_once_with("my_skill", ["search", "code"], 7.0)

    @pytest.mark.asyncio
    async def test_rollback_threshold(self, applier, settings):
        original = settings.thresholds.min_score
        mutations = await applier.apply([{"type": "adjust_threshold", "new_value": 8.0}])
        assert settings.thresholds.min_score == 8.0
        result = await applier.rollback(mutations[0].id)
        assert result is True
        assert settings.thresholds.min_score == original

    @pytest.mark.asyncio
    async def test_rollback_model(self, applier, settings):
        original = settings.models.triage
        mutations = await applier.apply([{"type": "escalate_model", "task_type": "triage", "to": "gemma4:26b"}])
        result = await applier.rollback(mutations[0].id)
        assert result is True
        assert settings.models.triage == original

    @pytest.mark.asyncio
    async def test_rollback_disable_tool(self, applier):
        mutations = await applier.apply([{"type": "disable_tool", "tool": "code"}])
        assert "code" in applier.disabled_tools
        await applier.rollback(mutations[0].id)
        assert "code" not in applier.disabled_tools

    @pytest.mark.asyncio
    async def test_max_mutations_per_cycle(self, applier, settings):
        settings.evolution.max_mutations_per_cycle = 2
        actions = [
            {"type": "adjust_threshold", "new_value": 5.0},
            {"type": "disable_tool", "tool": "a"},
            {"type": "disable_tool", "tool": "b"},
        ]
        mutations = await applier.apply(actions)
        assert len(mutations) == 2

    @pytest.mark.asyncio
    async def test_mutations_persisted(self, applier, settings):
        await applier.apply([{"type": "adjust_threshold", "new_value": 7.0}])
        log_path = Path(settings.evolution.mutations_log)
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action_type"] == "adjust_threshold"

    @pytest.mark.asyncio
    async def test_overrides_persisted_and_loaded(self, settings):
        applier1 = ActionApplier(settings)
        await applier1.apply([{"type": "adjust_threshold", "new_value": 7.0}])

        # Create a new applier that should load overrides
        s2 = Settings()
        s2.evolution.enabled = True
        s2.evolution.overrides_file = settings.evolution.overrides_file
        s2.evolution.mutations_log = settings.evolution.mutations_log
        applier2 = ActionApplier(s2)
        assert s2.thresholds.min_score == 7.0

    @pytest.mark.asyncio
    async def test_unknown_action_ignored(self, applier):
        mutations = await applier.apply([{"type": "unknown_action"}])
        assert len(mutations) == 0

    @pytest.mark.asyncio
    async def test_modify_source_appends(self, applier, tmp_path):
        target = tmp_path / "test_file.py"
        target.write_text("original content\n", encoding="utf-8")
        import agent.intelligence.action_applier as aa
        old_file = aa.__file__
        fake_path = tmp_path / "agent" / "intelligence" / "action_applier.py"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.touch()
        aa.__file__ = str(fake_path)
        try:
            mutations = await applier.apply([{
                "type": "modify_source",
                "target": "test_file.py",
                "patch_mode": "append",
                "new_value": "appended line",
                "reason": "test append",
            }])
            assert len(mutations) == 1
            result = target.read_text()
            assert "original content" in result
            assert "appended line" in result
            # Backup should exist
            assert target.with_suffix(".py.bak").exists()
        finally:
            aa.__file__ = old_file

    @pytest.mark.asyncio
    async def test_modify_source_default_mode_is_append(self, applier, tmp_path):
        """Actions without patch_mode default to append (backwards compat)."""
        target = tmp_path / "test_file.py"
        target.write_text("original\n", encoding="utf-8")
        import agent.intelligence.action_applier as aa
        old_file = aa.__file__
        fake_path = tmp_path / "agent" / "intelligence" / "action_applier.py"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.touch()
        aa.__file__ = str(fake_path)
        try:
            mutations = await applier.apply([{
                "type": "modify_source",
                "target": "test_file.py",
                "new_value": "added content",
                "reason": "test default",
            }])
            assert len(mutations) == 1
            result = target.read_text()
            assert "original" in result
            assert "added content" in result
        finally:
            aa.__file__ = old_file

    @pytest.mark.asyncio
    async def test_modify_source_skips_duplicate(self, applier, tmp_path):
        """Content already present is not appended again."""
        target = tmp_path / "test_file.py"
        target.write_text("original content\nalready here\n", encoding="utf-8")
        import agent.intelligence.action_applier as aa
        old_file = aa.__file__
        fake_path = tmp_path / "agent" / "intelligence" / "action_applier.py"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.touch()
        aa.__file__ = str(fake_path)
        try:
            mutations = await applier.apply([{
                "type": "modify_source",
                "target": "test_file.py",
                "patch_mode": "append",
                "new_value": "already here",
                "reason": "should skip",
            }])
            assert len(mutations) == 0
        finally:
            aa.__file__ = old_file

    @pytest.mark.asyncio
    async def test_modify_source_blocks_nonexistent(self, applier, tmp_path):
        """Patching a file that doesn't exist is blocked."""
        import agent.intelligence.action_applier as aa
        old_file = aa.__file__
        fake_path = tmp_path / "agent" / "intelligence" / "action_applier.py"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        fake_path.touch()
        aa.__file__ = str(fake_path)
        try:
            mutations = await applier.apply([{
                "type": "modify_source",
                "target": "nonexistent.py",
                "new_value": "content",
                "reason": "should fail",
            }])
            assert len(mutations) == 0
        finally:
            aa.__file__ = old_file

    @pytest.mark.asyncio
    async def test_modify_source_blocks_protected(self, applier, settings):
        settings.evolution.protected_files = ["agent/agent.py"]
        mutations = await applier.apply([{
            "type": "modify_source",
            "target": "agent/agent.py",
            "new_value": "hacked",
            "reason": "should be blocked",
        }])
        assert len(mutations) == 0
