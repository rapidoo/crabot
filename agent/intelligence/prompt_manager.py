"""Prompt manager — versioned prompt storage with A/B testing for evolution.

Allows the agent to propose, test, and promote prompt mutations
for planner, critic, and triage roles.

Prompt resolution order: file (agent/prompts/) > Neo4j > hardcoded default.
When a prompt is promoted, the corresponding file is updated automatically.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.memory.neo4j_client import MemoryClient

logger = logging.getLogger(__name__)

# Directory where prompt files live (tracked in git)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Default A/B test window: how many episodes before comparing scores
DEFAULT_EVAL_WINDOW = 10
# Minimum score delta to promote a candidate over the incumbent
MIN_PROMOTION_DELTA = 0.5


class PromptManager:
    """Manage versioned system prompts with A/B testing.

    Prompts are stored in Neo4j as PromptVersion nodes.
    Falls back to hardcoded defaults when Neo4j is unavailable.
    """

    def __init__(
        self,
        memory: MemoryClient | None = None,
        eval_window: int = DEFAULT_EVAL_WINDOW,
    ):
        self._memory = memory
        self._eval_window = eval_window
        # In-memory cache: role → {active_id, active_content, candidate_id, candidate_content}
        self._cache: dict[str, dict[str, Any]] = {}
        # Episode counter per role for A/B alternation
        self._episode_counter: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return self._memory is not None and getattr(self._memory, "available", False)

    def _read_prompt_file(self, role: str) -> str | None:
        """Read a prompt from the local file system (agent/prompts/<role>.md)."""
        path = PROMPTS_DIR / f"{role}.md"
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Failed to read prompt file %s: %s", path, exc)
        return None

    def _write_prompt_file(self, role: str, content: str) -> bool:
        """Write a prompt to the local file system (agent/prompts/<role>.md)."""
        path = PROMPTS_DIR / f"{role}.md"
        try:
            PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
            # Version backup
            if path.exists():
                bak = PROMPTS_DIR / f"{role}.md.bak"
                bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text(content.strip() + "\n", encoding="utf-8")
            logger.info("Prompt file updated: %s", path)
            return True
        except OSError as exc:
            logger.warning("Failed to write prompt file %s: %s", path, exc)
            return False

    async def get_prompt(self, role: str, default: str) -> str:
        """Get the current prompt for a role.

        Resolution order: file > Neo4j (A/B tested) > hardcoded default.
        If a candidate is in A/B testing, alternates between active and candidate.
        Returns the default if no prompt version exists.
        """
        # If A/B testing is active, use Neo4j path (candidate vs active)
        if self.available:
            if role not in self._cache:
                await self._load_cache(role)
            cached = self._cache.get(role)
            if cached and cached.get("candidate_content"):
                count = self._episode_counter.get(role, 0)
                self._episode_counter[role] = count + 1
                if count % 2 == 1:  # Odd episodes use candidate
                    return cached["candidate_content"]
                if cached.get("active_content"):
                    return cached["active_content"]

        # File-based prompt (highest priority outside A/B testing)
        file_prompt = self._read_prompt_file(role)
        if file_prompt:
            return file_prompt

        # Neo4j active prompt
        if self.available:
            cached = self._cache.get(role)
            if cached and cached.get("active_content"):
                return cached["active_content"]

        return default

    async def get_current_prompt_id(self, role: str) -> str | None:
        """Return the ID of the prompt that was last served (for score tracking)."""
        cached = self._cache.get(role)
        if not cached:
            return None
        count = self._episode_counter.get(role, 0)
        if cached.get("candidate_id") and (count - 1) % 2 == 1:
            return cached["candidate_id"]
        return cached.get("active_id")

    async def propose_mutation(
        self, role: str, new_content: str, reason: str = ""
    ) -> str | None:
        """Propose a new prompt version as a candidate for A/B testing.

        Returns the new prompt ID or None on failure.
        """
        if not self.available:
            return None

        # Determine next version number
        cached = self._cache.get(role, {})
        current_version = cached.get("active_version", 0)
        new_version = current_version + 1
        prompt_id = f"prompt_{role}_{uuid.uuid4().hex[:8]}"

        try:
            await self._memory.persist_prompt_version(
                prompt_id=prompt_id,
                role=role,
                version=new_version,
                content=new_content,
                is_active=False,  # Candidate, not yet promoted
            )
            # Update cache
            if role not in self._cache:
                self._cache[role] = {}
            self._cache[role]["candidate_id"] = prompt_id
            self._cache[role]["candidate_content"] = new_content
            self._cache[role]["candidate_version"] = new_version
            self._episode_counter[role] = 0  # Reset counter for new A/B test

            logger.info(
                "Prompt candidate proposed for %s (v%d): %s — %s",
                role, new_version, prompt_id, reason,
            )
            return prompt_id
        except Exception as exc:
            logger.warning("Failed to propose prompt mutation: %s", exc)
            return None

    async def record_score(self, role: str, score: float) -> None:
        """Record a score for the currently active prompt version."""
        prompt_id = await self.get_current_prompt_id(role)
        if prompt_id and self.available:
            try:
                await self._memory.update_prompt_stats(prompt_id, score)
            except Exception as exc:
                logger.warning("Failed to record prompt score: %s", exc)

    async def evaluate_and_promote(self, role: str) -> bool:
        """Check if a candidate prompt should be promoted.

        Returns True if promotion occurred.
        """
        if not self.available:
            return False

        cached = self._cache.get(role, {})
        candidate_id = cached.get("candidate_id")
        active_id = cached.get("active_id")
        if not candidate_id or not active_id:
            return False

        count = self._episode_counter.get(role, 0)
        if count < self._eval_window:
            return False  # Not enough episodes yet

        try:
            active = await self._memory.get_active_prompt(role)
            candidate = await self._memory.get_candidate_prompt(role)

            if not active or not candidate:
                return False

            active_score = active.get("score_avg", 0.0)
            candidate_score = candidate.get("score_avg", 0.0)
            active_uses = active.get("usage_count", 0)
            candidate_uses = candidate.get("usage_count", 0)

            # Both need enough data
            min_uses = self._eval_window // 2
            if active_uses < min_uses or candidate_uses < min_uses:
                return False

            delta = candidate_score - active_score
            if delta >= MIN_PROMOTION_DELTA:
                await self._memory.promote_prompt(role, candidate_id)
                promoted_content = cached.get("candidate_content", "")
                # Update cache
                self._cache[role] = {
                    "active_id": candidate_id,
                    "active_content": promoted_content,
                    "active_version": cached.get("candidate_version", 0),
                    "candidate_id": None,
                    "candidate_content": None,
                }
                self._episode_counter[role] = 0
                # Write promoted prompt to file (visible in git diff)
                self._write_prompt_file(role, promoted_content)
                logger.info(
                    "Prompt promoted for %s: %s (delta=+%.2f, %.1f → %.1f)",
                    role, candidate_id, delta, active_score, candidate_score,
                )
                return True
            elif count >= self._eval_window * 2:
                # Candidate underperforming after extended test — discard
                self._cache[role].pop("candidate_id", None)
                self._cache[role].pop("candidate_content", None)
                self._episode_counter[role] = 0
                logger.info(
                    "Prompt candidate discarded for %s (delta=%.2f)",
                    role, delta,
                )

        except Exception as exc:
            logger.warning("Prompt evaluation failed: %s", exc)

        return False

    async def rollback_prompt(self, role: str) -> bool:
        """Revert to the previous prompt version for a role."""
        if not self.available:
            return False

        # Clear candidate if any
        cached = self._cache.get(role, {})
        if cached.get("candidate_id"):
            self._cache[role].pop("candidate_id", None)
            self._cache[role].pop("candidate_content", None)
            self._episode_counter[role] = 0
            logger.info("Rolled back candidate prompt for %s", role)
            return True
        return False

    async def _load_cache(self, role: str) -> None:
        """Load prompt versions from Neo4j into the in-memory cache."""
        if not self.available:
            return
        try:
            active = await self._memory.get_active_prompt(role)
            candidate = await self._memory.get_candidate_prompt(role)
            self._cache[role] = {
                "active_id": active.get("id") if active else None,
                "active_content": active.get("content") if active else None,
                "active_version": active.get("version", 0) if active else 0,
                "candidate_id": candidate.get("id") if candidate else None,
                "candidate_content": candidate.get("content") if candidate else None,
                "candidate_version": candidate.get("version", 0) if candidate else 0,
            }
        except Exception as exc:
            logger.warning("Failed to load prompt cache for %s: %s", role, exc)
            self._cache[role] = {}
