"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from agent.config import Settings, SamplingParams, ModelsConfig, load_settings, MODEL_PRESETS


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
        assert s.thresholds.min_score == 6.5
        assert s.thinking.planner is True
        assert s.thinking.executor is False

    def test_override(self):
        s = Settings(thresholds={"min_score": 7.0, "max_retries": 5})
        assert s.thresholds.min_score == 7.0
        assert s.thresholds.max_retries == 5


class TestModelPresets:
    def test_gemma4_preset(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "GEMMA4")
        s = load_settings(Path("/nonexistent.yaml"))
        assert s.models.triage == "gemma4:e4b"
        assert s.models.planner == "gemma4:26b"
        assert s.models.executor == "gemma4:26b"
        assert s.models.executor_draft == "gemma4:e4b"
        assert s.models.critic == "gemma4:31b"
        assert s.models.critic_light == "gemma4:26b"

    def test_mistral_preset(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "MISTRAL")
        s = load_settings(Path("/nonexistent.yaml"))
        assert s.models.triage == "ministral-3:8b"
        assert s.models.planner == "mistral-small3.2"
        assert s.models.executor == "mistral-small3.2"
        assert s.models.executor_draft == "ministral-3:3b"
        assert s.models.critic == "mistral-small4"
        assert s.models.critic_light == "mistral-small3.2"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "mistral")
        s = load_settings(Path("/nonexistent.yaml"))
        assert s.models.triage == "ministral-3:8b"

    def test_unknown_preset_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "LLAMA")
        with pytest.raises(ValueError, match="Unknown MODEL_NAME"):
            load_settings(Path("/nonexistent.yaml"))

    def test_default_preset_without_env(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        s = load_settings(Path("/nonexistent.yaml"))
        # Default is GEMMA4
        assert s.models.triage == "gemma4:e4b"

    def test_yaml_overrides_preset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_NAME", "MISTRAL")
        config = {"models": {"triage": "custom-model:7b"}}
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(config))

        s = load_settings(p)
        # triage overridden by YAML
        assert s.models.triage == "custom-model:7b"
        # rest from Mistral preset
        assert s.models.planner == "mistral-small3.2"


class TestLoadSettings:
    def test_load_from_yaml(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        config = {
            "models": {"triage": "gemma4:e2b"},
            "thresholds": {"min_score": 8.0},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(config))

        s = load_settings(p)
        assert s.models.triage == "gemma4:e2b"
        assert s.thresholds.min_score == 8.0
        # Unset model values come from default preset (GEMMA4)
        assert s.models.planner == "gemma4:26b"

    def test_load_missing_file_returns_defaults(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        s = load_settings(tmp_path / "nonexistent.yaml")
        assert s.models.triage == "gemma4:e4b"

    def test_load_empty_file(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        p = tmp_path / "config.yaml"
        p.write_text("")
        s = load_settings(p)
        assert s.models.triage == "gemma4:e4b"

    def test_load_real_config(self, monkeypatch):
        """Load the actual config.yaml shipped with the project."""
        monkeypatch.delenv("MODEL_NAME", raising=False)
        s = load_settings(Path(__file__).parent.parent / "agent" / "config.yaml")
        assert s.models.ollama_base_url == "http://localhost:11434"
        assert s.sampling.critic.temperature == 0.3
        assert s.skeleton.token_threshold == 500
