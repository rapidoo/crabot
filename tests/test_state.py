"""Tests for StateManager — cursor recovery and atomic persistence."""

import json

import pytest

from agent.infra.state import StateManager
from agent.schemas import Plan, Step


def _make_plan(n_steps: int = 3) -> Plan:
    steps = [
        Step(id=i + 1, tool="none", input=f"step {i+1}", expected_output=f"output {i+1}")
        for i in range(n_steps)
    ]
    return Plan(goal="Test plan", steps=steps)


class TestStateManager:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        sm = StateManager(state_file)
        plan = _make_plan()

        sm.save_plan(plan)
        assert state_file.exists()

        sm2 = StateManager(state_file)
        loaded = sm2.load_state()
        assert loaded is not None
        loaded_plan, cursor = loaded
        assert loaded_plan.goal == "Test plan"
        assert len(loaded_plan.steps) == 3
        assert cursor == 0

    def test_advance_cursor(self, tmp_path):
        state_file = tmp_path / "state.json"
        sm = StateManager(state_file)
        plan = _make_plan()

        sm.save_plan(plan)
        sm.advance_cursor(1)
        sm.advance_cursor(2)

        sm2 = StateManager(state_file)
        loaded = sm2.load_state()
        assert loaded is not None
        _, cursor = loaded
        assert cursor == 2

    def test_clear(self, tmp_path):
        state_file = tmp_path / "state.json"
        sm = StateManager(state_file)
        sm.save_plan(_make_plan())
        assert state_file.exists()

        sm.clear()
        assert not state_file.exists()
        assert sm.plan is None
        assert sm.cursor == 0

    def test_has_pending_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        sm = StateManager(state_file)

        assert not sm.has_pending_state
        sm.save_plan(_make_plan())
        assert sm.has_pending_state

    def test_load_nonexistent(self, tmp_path):
        sm = StateManager(tmp_path / "nope.json")
        assert sm.load_state() is None

    def test_corrupted_state_resets(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json{{{")

        sm = StateManager(state_file)
        assert sm.load_state() is None
        assert not state_file.exists()  # corrupted file removed

    def test_atomic_write(self, tmp_path):
        state_file = tmp_path / "state.json"
        sm = StateManager(state_file)
        plan = _make_plan()

        sm.save_plan(plan)

        # Verify actual file content is valid JSON
        data = json.loads(state_file.read_text())
        assert data["plan"]["goal"] == "Test plan"
        assert data["cursor"] == 0

        # No .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_creates_parent_dirs(self, tmp_path):
        state_file = tmp_path / "sub" / "dir" / "state.json"
        sm = StateManager(state_file)
        sm.save_plan(_make_plan())
        assert state_file.exists()
