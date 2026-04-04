"""Tests for ModelRouter."""

from agent.config import Settings, SamplingParams
from agent.models.router import ModelRouter


class TestModelRouter:
    def setup_method(self):
        self.router = ModelRouter(Settings())

    def test_select_triage(self):
        assert self.router.select("triage") == "gemma4:e4b"

    def test_select_planner(self):
        assert self.router.select("planner") == "gemma4:26b"

    def test_select_executor(self):
        assert self.router.select("executor") == "gemma4:26b"

    def test_select_critic_default_uses_light(self):
        # Without high_stakes, critic uses the lighter model
        assert self.router.select("critic") == "gemma4:26b"

    def test_select_critic_high_stakes(self):
        assert self.router.select("critic", high_stakes=True) == "gemma4:31b"

    def test_select_executor_draft(self):
        assert self.router.select("executor_draft") == "gemma4:e4b"

    def test_select_unknown_role_falls_back(self):
        # Unknown roles fall back to executor model
        assert self.router.select("unknown_role") == "gemma4:26b"

    def test_thinking_enabled(self):
        assert self.router.thinking_enabled("planner") is True
        assert self.router.thinking_enabled("critic") is True
        assert self.router.thinking_enabled("triage") is False
        assert self.router.thinking_enabled("executor") is False

    def test_sampling_planner(self):
        s = self.router.sampling("planner")
        assert isinstance(s, SamplingParams)
        assert s.temperature == 0.7

    def test_sampling_critic(self):
        s = self.router.sampling("critic")
        assert s.temperature == 0.3
        assert s.top_k == 32

    def test_sampling_default_fallback(self):
        s = self.router.sampling("unknown")
        assert s.temperature == 1.0
