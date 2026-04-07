"""Configuration loader — reads config.yaml into typed pydantic models."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel


class SamplingParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


MODEL_PRESETS: dict[str, dict[str, str]] = {
    "GEMMA4": {
        "triage": "gemma4:e4b",
        "planner": "gemma4:26b",
        "executor": "gemma4:26b",
        "executor_draft": "gemma4:e4b",
        "critic": "gemma4:31b",
        "critic_light": "gemma4:26b",
    },
    "MISTRAL": {
        "triage": "ministral-3:8b",
        "planner": "mistral-small3.2",
        "executor": "mistral-small3.2",
        "executor_draft": "ministral-3:3b",
        "critic": "mistral-small3.2",
        "critic_light": "mistral-small3.2",
    },
}

_DEFAULT_PRESET = "GEMMA4"


def _get_model_preset() -> dict[str, str]:
    """Return the model preset based on MODEL_NAME env var."""
    name = os.environ.get("MODEL_NAME", _DEFAULT_PRESET).upper()
    if name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown MODEL_NAME={name!r}. Valid presets: {list(MODEL_PRESETS)}"
        )
    return MODEL_PRESETS[name]


def _default_models() -> dict[str, str]:
    preset = _get_model_preset()
    return {**preset, "ollama_base_url": "http://localhost:11434"}


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
    neo4j_password: str = ""
    context_k: int = 5
    write_threshold: float = 5.0
    read_threshold: float = 6.5
    entity_extraction: bool = True
    compress_after_days: int = 30
    compress_min_keep: int = 50
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    skills_cache_ttl: float = 300.0


class ContextConfig(BaseModel):
    max_tokens_per_call: int = 32768
    skeleton_threshold: int = 500
    trace_in_context: bool = True
    compression_threshold: float = 0.50
    compression_protect_last_n: int = 4


class ApprovalConfig(BaseModel):
    enabled: bool = True
    dangerous_tools: list[str] = ["code", "tool_create"]


class TruncationConfig(BaseModel):
    enabled: bool = True
    max_chars: int = 50_000
    tail_chars: int = 2_000


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


class LessonConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.85
    max_in_context: int = 5
    decay_days: int = 90


class EvolutionConfig(BaseModel):
    enabled: bool = False
    max_mutations_per_cycle: int = 3
    overrides_file: str = "./state/overrides.yaml"
    mutations_log: str = "./state/mutations.jsonl"
    protected_roles: list[str] = ["critic"]
    tools_dir: str = "agent/tools/custom"
    trusted_tools: bool = False
    protected_files: list[str] = [
        "agent/agent.py",
        "agent/config.py",
    ]


class SupervisorConfig(BaseModel):
    socket_path: str = "/tmp/nano-agent.sock"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    worker_restart_delay: int = 2
    max_restart_attempts: int = 5
    drain_timeout: int = 30


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
    evolution: EvolutionConfig = EvolutionConfig()
    lessons: LessonConfig = LessonConfig()
    supervisor: SupervisorConfig = SupervisorConfig()
    approval: ApprovalConfig = ApprovalConfig()
    truncation: TruncationConfig = TruncationConfig()


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from a YAML file. Falls back to defaults if file is missing.

    Model names are resolved from the MODEL_NAME env var preset,
    then overridden by any explicit values in the YAML file.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Apply model preset defaults, let YAML overrides win
    preset_models = _default_models()
    yaml_models = data.get("models", {})
    data["models"] = {**preset_models, **yaml_models}

    return Settings.model_validate(data)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this from application code."""
    return load_settings()
