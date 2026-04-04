"""Model router — selects the optimal Gemma 4 variant per role and context."""

from __future__ import annotations

from agent.config import Settings, get_settings


class ModelRouter:
    """Select model + sampling + thinking config per pipeline role."""

    def __init__(self, settings: Settings | None = None):
        self._s = settings or get_settings()

    def select(self, role: str, *, high_stakes: bool = False) -> str:
        """Return the Ollama model name for the given role.

        Roles: triage, planner, executor, executor_draft, critic, critic_light.
        If high_stakes=True and role is critic, always use the full critic model.
        """
        if role == "critic" and not high_stakes:
            return self._s.models.critic_light
        return getattr(self._s.models, role, self._s.models.executor)

    def thinking_enabled(self, role: str) -> bool:
        """Whether thinking mode should be on for this role."""
        return getattr(self._s.thinking, role, False)

    def sampling(self, role: str):
        """Return the SamplingParams for the given role."""
        return getattr(self._s.sampling, role, self._s.sampling.default)
