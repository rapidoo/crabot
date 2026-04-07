"""Action applier — applies reflection actions to runtime settings with rollback support."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
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
            "modify_source": self._apply_modify_source,
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

    async def _apply_modify_source(
        self, action: dict[str, Any], reflection_id: str | None
    ) -> Mutation | None:
        """Patch a source file (append/prepend/insert_after). Full replacement is blocked."""
        target_path = action.get("target", "")
        content = action.get("new_value", action.get("content", ""))
        if not target_path or not content:
            return None

        patch_mode = action.get("patch_mode", "append")
        if patch_mode not in ("append", "prepend", "insert_after"):
            logger.warning("modify_source blocked: invalid patch_mode '%s'", patch_mode)
            return None

        # Resolve path relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        full_path = (repo_root / target_path).resolve()

        # Safety: must stay within the repo
        if not str(full_path).startswith(str(repo_root)):
            logger.warning("modify_source blocked: path escapes repo: %s", target_path)
            return None

        # Safety: check protected files
        for protected in self._evo.protected_files:
            protected_abs = (repo_root / protected).resolve()
            if full_path == protected_abs:
                logger.warning("modify_source blocked: protected file: %s", target_path)
                return None

        # File must already exist for patching
        if not full_path.exists():
            logger.warning("modify_source blocked: file does not exist: %s", target_path)
            return None

        prev_content = full_path.read_text(encoding="utf-8")

        # Deduplicate: skip if content already present
        if content.strip() in prev_content:
            logger.info("modify_source skipped: content already present in %s", target_path)
            return None

        # Apply patch
        if patch_mode == "append":
            patched = prev_content.rstrip() + "\n\n" + content.strip() + "\n"
        elif patch_mode == "prepend":
            patched = content.strip() + "\n\n" + prev_content.lstrip()
        elif patch_mode == "insert_after":
            marker = action.get("marker", "")
            if not marker or marker not in prev_content:
                logger.warning("modify_source blocked: marker not found in %s: '%s'",
                               target_path, marker[:80])
                return None
            patched = prev_content.replace(marker, marker + "\n" + content.strip(), 1)

        # Safety: patch must not shrink the file significantly (catch accidental truncation)
        if len(patched) < len(prev_content) * 0.8:
            logger.warning("modify_source blocked: patch would shrink %s by >20%%", target_path)
            return None

        # Create .bak version and write
        bak_path = full_path.with_suffix(full_path.suffix + ".bak")
        shutil.copy2(full_path, bak_path)
        full_path.write_text(patched, encoding="utf-8")
        logger.info("Source patched (%s): %s", patch_mode, target_path)

        return self._make_mutation(
            "modify_source", target_path,
            hashlib.sha256(prev_content.encode()).hexdigest()[:12],
            hashlib.sha256(patched.encode()).hexdigest()[:12],
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
        elif mutation.action_type == "modify_source":
            repo_root = Path(__file__).resolve().parent.parent.parent
            full_path = repo_root / mutation.target
            bak_path = full_path.with_suffix(full_path.suffix + ".bak")
            if bak_path.exists():
                shutil.copy2(bak_path, full_path)
                logger.info("Restored %s from backup", mutation.target)

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
