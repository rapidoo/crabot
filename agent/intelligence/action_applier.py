"""Action applier — applies reflection actions to runtime settings with rollback support."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.config import Settings
from agent.schemas import Mutation

logger = logging.getLogger(__name__)


class ActionApplier:
    """Apply reflection actions to the running agent and persist overrides.

    Supports rollback via a mutations journal (JSONL).
    """

    def __init__(self, settings: Settings, memory: Any = None):
        self._settings = settings
        self._memory = memory
        self._evo = settings.evolution
        self._overrides_path = Path(self._evo.overrides_file)
        self._mutations_path = Path(self._evo.mutations_log)
        self._disabled_tools: set[str] = set()
        self._mutations: list[Mutation] = []
        self._load_overrides()

    @property
    def disabled_tools(self) -> set[str]:
        return set(self._disabled_tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply(
        self,
        actions: list[dict[str, Any]],
        reflection_id: str | None = None,
    ) -> list[Mutation]:
        """Apply a batch of reflection actions. Returns the list of applied mutations."""
        if not self._evo.enabled:
            logger.info("Evolution disabled — skipping %d actions", len(actions))
            return []

        applied: list[Mutation] = []
        for action in actions[: self._evo.max_mutations_per_cycle]:
            mutation = await self._apply_one(action, reflection_id)
            if mutation:
                applied.append(mutation)
                self._mutations.append(mutation)
                self._persist_mutation(mutation)

        if applied:
            self._save_overrides()

        return applied

    async def rollback(self, mutation_id: str) -> bool:
        """Rollback a specific mutation by restoring its previous value."""
        for mutation in reversed(self._mutations):
            if mutation.id == mutation_id and not mutation.rolled_back:
                self._restore(mutation)
                mutation.rolled_back = True
                self._persist_mutation(mutation)
                self._save_overrides()
                logger.info("Rolled back mutation %s (%s)", mutation_id, mutation.action_type)
                return True
        logger.warning("Mutation %s not found or already rolled back", mutation_id)
        return False

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _apply_one(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        action_type = action.get("type", "")
        handler = {
            "adjust_threshold": self._apply_threshold,
            "escalate_model": self._apply_model,
            "disable_tool": self._apply_disable_tool,
            "create_skill": self._apply_create_skill,
        }.get(action_type)

        if handler is None:
            logger.warning("Unknown action type: %s", action_type)
            return None

        try:
            return await handler(action, reflection_id)
        except Exception as exc:
            logger.error("Failed to apply %s: %s", action_type, exc)
            return None

    async def _apply_threshold(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        new_val = action.get("new_value")
        if new_val is None or not isinstance(new_val, (int, float)):
            return None
        new_val = max(4.0, min(9.0, float(new_val)))

        prev = self._settings.thresholds.min_score
        self._settings.thresholds.min_score = new_val
        logger.info("Threshold adjusted: %.1f → %.1f", prev, new_val)

        return self._make_mutation(
            "adjust_threshold", "thresholds.min_score",
            prev, new_val, action.get("reason", ""), reflection_id,
        )

    async def _apply_model(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        role = action.get("task_type", action.get("role", ""))
        new_model = action.get("to", "")
        if not role or not new_model:
            return None

        if role in self._evo.protected_roles:
            logger.warning("Cannot modify protected role: %s", role)
            return None

        prev = getattr(self._settings.models, role, None)
        if prev is None:
            return None

        setattr(self._settings.models, role, new_model)
        logger.info("Model escalated: %s %s → %s", role, prev, new_model)

        return self._make_mutation(
            "escalate_model", f"models.{role}",
            prev, new_model, action.get("reason", ""), reflection_id,
        )

    async def _apply_disable_tool(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        tool_name = action.get("tool", "")
        if not tool_name:
            return None

        if tool_name in self._disabled_tools:
            return None

        self._disabled_tools.add(tool_name)
        from agent.tools.registry import disable_tool
        disable_tool(tool_name)
        logger.info("Tool disabled: %s", tool_name)

        return self._make_mutation(
            "disable_tool", f"tools.{tool_name}",
            "enabled", "disabled",
            action.get("reason", ""), reflection_id,
        )

    async def _apply_create_skill(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        name = action.get("name", "")
        chain = action.get("tool_chain", [])
        if not name or not chain:
            return None

        if self._memory and getattr(self._memory, "available", False):
            try:
                await self._memory.persist_skill(name, chain, 7.0)
            except Exception as exc:
                logger.warning("Skill creation failed: %s", exc)
                return None

        logger.info("Skill created: %s (chain=%s)", name, chain)
        return self._make_mutation(
            "create_skill", f"skills.{name}",
            None, chain,
            action.get("reason", ""), reflection_id,
        )

    # ------------------------------------------------------------------
    # Restore (rollback helper)
    # ------------------------------------------------------------------

    def _restore(self, mutation: Mutation) -> None:
        if mutation.action_type == "adjust_threshold":
            self._settings.thresholds.min_score = float(mutation.previous_value)
        elif mutation.action_type == "escalate_model":
            role = mutation.target.replace("models.", "")
            setattr(self._settings.models, role, mutation.previous_value)
        elif mutation.action_type == "disable_tool":
            tool_name = mutation.target.replace("tools.", "")
            self._disabled_tools.discard(tool_name)
            from agent.tools.registry import enable_tool
            enable_tool(tool_name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _make_mutation(
        self, action_type: str, target: str,
        prev: Any, new: Any, reason: str, reflection_id: str | None,
    ) -> Mutation:
        return Mutation(
            id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_type=action_type,
            target=target,
            previous_value=prev,
            new_value=new,
            reason=reason,
            reflection_id=reflection_id,
        )

    def _persist_mutation(self, mutation: Mutation) -> None:
        self._mutations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._mutations_path, "a") as f:
            f.write(mutation.model_dump_json() + "\n")

    def _save_overrides(self) -> None:
        """Persist current runtime overrides to YAML for next startup."""
        overrides: dict[str, Any] = {
            "thresholds": {"min_score": self._settings.thresholds.min_score},
            "disabled_tools": sorted(self._disabled_tools),
        }
        # Capture model overrides
        model_overrides = {}
        for role in ("triage", "planner", "executor", "executor_draft"):
            model_overrides[role] = getattr(self._settings.models, role)
        overrides["models"] = model_overrides

        self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._overrides_path, "w") as f:
            yaml.dump(overrides, f, default_flow_style=False)

    def _load_overrides(self) -> None:
        """Load persisted overrides from previous sessions."""
        if not self._overrides_path.exists():
            return
        try:
            with open(self._overrides_path) as f:
                data = yaml.safe_load(f) or {}

            if "thresholds" in data:
                t = data["thresholds"]
                if "min_score" in t:
                    self._settings.thresholds.min_score = float(t["min_score"])

            if "disabled_tools" in data:
                self._disabled_tools = set(data["disabled_tools"])
                from agent.tools.registry import disable_tool
                for tool_name in self._disabled_tools:
                    disable_tool(tool_name)

            if "models" in data:
                for role, model in data["models"].items():
                    if role not in self._evo.protected_roles and hasattr(self._settings.models, role):
                        setattr(self._settings.models, role, model)

            logger.info("Loaded %d overrides from %s", len(data), self._overrides_path)
        except Exception as exc:
            logger.warning("Failed to load overrides: %s", exc)
