"""Configuration loader — reads config.yaml into typed pydantic models."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel


class SamplingParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


class ModelsConfig(BaseModel):
    triage: str = "gemma4:e4b"
    planner: str = "gemma4:26b"
    executor: str = "gemma4:26b"
    executor_draft: str = "gemma4:e4b"
    critic: str = "gemma4:31b"
    critic_light: str = "gemma4:26b"
    ollama_base_url: str = "http://localhost:11434"


class SamplingConfig(BaseModel):
    default: SamplingParams = SamplingParams()
    planner: SamplingParams = SamplingParams(temperature=0.7)
    critic: SamplingParams = SamplingParams(temperature=0.3, top_p=0.9, top_k=32)


class ThinkingConfig(BaseModel):
    triage: bool = False
    planner: bool = True
    executor: bool = False
    critic: bool = True


class ThresholdsConfig(BaseModel):
    min_score: float = 6.5
    max_retries: int = 3
    parallel_steps: bool = True


class SkeletonConfig(BaseModel):
    enabled: bool = True
    token_threshold: int = 500


class SpeculativeConfig(BaseModel):
    enabled: bool = True
    max_input_tokens: int = 200
    tools_bypass: list[str] = ["code", "search"]


class VisionConfig(BaseModel):
    token_budget_ocr: int = 1120
    token_budget_default: int = 280
    token_budget_fast: int = 70


class RecoveryConfig(BaseModel):
    persist_cursor: bool = True
    state_file: str = "./state/agent_state.json"
    atomic_write: bool = True


class MemoryConfig(BaseModel):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    context_k: int = 5
    write_threshold: float = 5.0
    entity_extraction: bool = True


class ContextConfig(BaseModel):
    max_tokens_per_call: int = 32768
    skeleton_threshold: int = 500
    trace_in_context: bool = True


class TelegramConfig(BaseModel):
    bot_token: str = ""
    allowed_users: list[int] = []
    max_message_length: int = 4000


class DaemonConfig(BaseModel):
    max_concurrent: int = 3
    request_timeout: int = 300


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str = "./logs/agent_trace.jsonl"


class CronTaskConfig(BaseModel):
    name: str
    schedule: str  # cron expression or interval in seconds
    prompt: str
    schedule_type: str = "cron"  # "cron" | "interval" | "once"


class SchedulerConfig(BaseModel):
    enabled: bool = False
    cron_tasks: list[CronTaskConfig] = []
    watch_paths: list[str] = []
    self_schedule: bool = True


class FeedbackConfig(BaseModel):
    enabled: bool = True
    good_score: float = 9.0
    bad_score: float = 2.0


class AdaptiveConfig(BaseModel):
    enabled: bool = True
    min_samples: int = 20
    threshold_method: str = "mean_minus_std"


class ReflectionConfig(BaseModel):
    enabled: bool = False
    last_n_episodes: int = 50
    auto_apply: bool = False


class Settings(BaseModel):
    models: ModelsConfig = ModelsConfig()
    sampling: SamplingConfig = SamplingConfig()
    thinking: ThinkingConfig = ThinkingConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    skeleton: SkeletonConfig = SkeletonConfig()
    speculative: SpeculativeConfig = SpeculativeConfig()
    vision: VisionConfig = VisionConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    memory: MemoryConfig = MemoryConfig()
    context: ContextConfig = ContextConfig()
    telegram: TelegramConfig = TelegramConfig()
    daemon: DaemonConfig = DaemonConfig()
    logging: LoggingConfig = LoggingConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()
    reflection: ReflectionConfig = ReflectionConfig()


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from a YAML file. Falls back to defaults if file is missing."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Settings.model_validate(data)
    return Settings()


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this from application code."""
    return load_settings()
