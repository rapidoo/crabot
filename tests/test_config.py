"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from agent.config import Settings, SamplingParams, load_settings


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 1.0
        assert p.top_p == 0.95
        assert p.top_k == 64

    def test_custom(self):
        p = SamplingParams(temperature=0.3, top_p=0.9, top_k=32)
        assert p.temperature == 0.3


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.models.triage == "gemma4:e4b"
        assert s.models.planner == "gemma4:26b"
        assert s.models.critic == "gemma4:31b"
        assert s.thresholds.min_score == 6.5
        assert s.thinking.planner is True
        assert s.thinking.executor is False

    def test_override(self):
        s = Settings(thresholds={"min_score": 7.0, "max_retries": 5})
        assert s.thresholds.min_score == 7.0
        assert s.thresholds.max_retries == 5


class TestLoadSettings:
    def test_load_from_yaml(self, tmp_path: Path):
        config = {
            "models": {"triage": "gemma4:e2b"},
            "thresholds": {"min_score": 8.0},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(config))

        s = load_settings(p)
        assert s.models.triage == "gemma4:e2b"
        assert s.thresholds.min_score == 8.0
        # Unset values keep defaults
        assert s.models.planner == "gemma4:26b"

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        s = load_settings(tmp_path / "nonexistent.yaml")
        assert s.models.triage == "gemma4:e4b"

    def test_load_empty_file(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text("")
        s = load_settings(p)
        assert s == Settings()

    def test_load_real_config(self):
        """Load the actual config.yaml shipped with the project."""
        s = load_settings(Path(__file__).parent.parent / "agent" / "config.yaml")
        assert s.models.ollama_base_url == "http://localhost:11434"
        assert s.sampling.critic.temperature == 0.3
        assert s.skeleton.token_threshold == 500
