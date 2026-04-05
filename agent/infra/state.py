"""Cursor recovery — persist plan execution progress for crash resilience."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from agent.schemas import Plan

logger = logging.getLogger(__name__)


class StateManager:
    """Persist and recover plan execution state.

    Uses atomic writes (.tmp + rename) to prevent corruption.
    """

    def __init__(self, state_file: str | Path):
        self._path = Path(state_file)
        self._cursor: int = 0
        self._plan: Plan | None = None
        self._user_input: str = ""

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def plan(self) -> Plan | None:
        return self._plan

    @property
    def has_pending_state(self) -> bool:
        """Check if there's a saved state to resume from."""
        return self._path.exists()

    def save_plan(self, plan: Plan, user_input: str = "") -> None:
        """Save a new plan and reset cursor to 0."""
        self._plan = plan
        self._cursor = 0
        self._user_input = user_input
        self._persist()
        logger.info("State: saved plan with %d steps", len(plan.steps))

    def advance_cursor(self, step_id: int) -> None:
        """Mark a step as completed and persist."""
        self._cursor = step_id
        self._persist()
        logger.debug("State: cursor advanced to step %d", step_id)

    def load_state(self) -> tuple[Plan, int, str] | None:
        """Load saved state. Returns (plan, resume_from_step_id, user_input) or None."""
        if not self._path.exists():
            return None

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            plan = Plan.model_validate(data["plan"])
            cursor = data.get("cursor", 0)
            user_input = data.get("user_input", "")
            self._plan = plan
            self._cursor = cursor
            self._user_input = user_input
            logger.info(
                "State: loaded plan (%d steps), resuming from step %d",
                len(plan.steps), cursor + 1,
            )
            return plan, cursor, user_input
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("State: corrupted state file, resetting: %s", exc)
            self.clear()
            return None

    def clear(self) -> None:
        """Remove state file."""
        if self._path.exists():
            self._path.unlink()
        self._plan = None
        self._cursor = 0
        logger.debug("State: cleared")

    def _persist(self) -> None:
        """Atomic write: write to .tmp then rename."""
        if self._plan is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "plan": self._plan.model_dump(),
            "cursor": self._cursor,
            "user_input": self._user_input,
        }

        # Write to temp file in same directory, then atomic rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f)
            os.rename(tmp_path, str(self._path))
        except Exception:
            # Clean up tmp on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
