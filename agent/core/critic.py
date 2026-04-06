"""Critic — scores executor outputs and triggers retries when quality is low."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from agent.config import get_settings
from agent.infra.metrics import MetricsCollector
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.schemas import Plan, Step, StepResult, CriticScore, ScoredResult

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """\
You are a strict quality evaluator. Score the following output against
the expected result. Return JSON ONLY. No prose.

Criteria (score 0-10 each):
  - completeness : does the output fully cover the expected_output?
  - accuracy     : no detectable factual hallucination?
  - format       : does the output match the requested format?
  - coherence    : consistent with previous steps context?

Output format (strict):
{
  "scores": {
    "completeness": X,
    "accuracy": X,
    "format": X,
    "coherence": X
  },
  "final_score": X.X,
  "retry": true|false,
  "reason": "one sentence explanation if retry=true"
}

Retry threshold: final_score < 6.5
Max retries per step: 3"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_critic_score(raw: str) -> CriticScore:
    """Parse raw LLM output into a CriticScore."""
    text = raw.strip()
    if not text.startswith("{"):
        m = _JSON_OBJECT_RE.search(text)
        if m:
            text = m.group(0)
    data = json.loads(text)
    return CriticScore.model_validate(data)


class AdaptiveThreshold:
    """Critic threshold calibrated on past score distribution."""

    def __init__(
        self,
        default: float = 6.5,
        min_samples: int = 20,
        trace_file: str | None = None,
    ):
        self.default = default
        self.min_samples = min_samples
        self._metrics = MetricsCollector(trace_file) if trace_file else None

    def get(self, task_type: str = "default") -> float:
        """Return the adaptive threshold, or default if insufficient data."""
        if self._metrics is None:
            return self.default
        scores = self._metrics.get_scores(task_type, limit=50)
        if len(scores) < self.min_samples:
            return self.default
        avg = sum(scores) / len(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        # Reject the bottom quartile — adapted to the model's actual level
        return max(round(avg - std, 1), 4.0)


class Critic:
    """Evaluates step results and triggers retries on low scores."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        router: ModelRouter | None = None,
        executor: object | None = None,
        prompt_manager: object | None = None,
    ):
        settings = get_settings()
        self._client = client or OllamaClient(
            base_url=settings.models.ollama_base_url
        )
        self._router = router or ModelRouter(settings)
        self._executor = executor
        self._prompt_manager = prompt_manager
        self._settings = settings
        adaptive_cfg = getattr(settings, "adaptive", None)
        if adaptive_cfg and adaptive_cfg.enabled:
            self._threshold = AdaptiveThreshold(
                default=settings.thresholds.min_score,
                min_samples=adaptive_cfg.min_samples,
                trace_file=settings.logging.trace_file,
            )
        else:
            self._threshold = AdaptiveThreshold(
                default=settings.thresholds.min_score
            )

    async def evaluate(
        self, plan: Plan, results: list[StepResult]
    ) -> list[ScoredResult]:
        """Score each step result and retry if below threshold."""
        scored: list[ScoredResult] = []
        max_retries = self._settings.thresholds.max_retries
        min_score = self._threshold.get()

        for step, result in zip(plan.steps, results):
            current_result = result

            for attempt in range(max_retries):
                score = await self._score_step(step, current_result, high_stakes=(attempt > 0))
                logger.info(
                    "Step %d score: %.1f (attempt %d/%d)",
                    step.id, score.final_score, attempt + 1, max_retries,
                )

                if score.final_score >= min_score:
                    break

                # Retry if executor is available
                if self._executor is not None and attempt < max_retries - 1:
                    logger.info(
                        "Step %d retry: %s", step.id, score.reason
                    )
                    current_result = await self._executor.execute_step(step)
                else:
                    break

            scored.append(
                ScoredResult(step=step, result=current_result, score=score)
            )

        return scored

    async def _get_critic_prompt(self) -> str:
        """Get the critic prompt, checking PromptManager for an evolved version."""
        if self._prompt_manager and hasattr(self._prompt_manager, "get_prompt"):
            try:
                return await self._prompt_manager.get_prompt("critic", CRITIC_SYSTEM_PROMPT)
            except Exception:
                pass
        return CRITIC_SYSTEM_PROMPT

    async def _score_step(self, step: Step, result: StepResult, *, high_stakes: bool = False) -> CriticScore:
        """Score a single step result."""
        model = self._router.select("critic", high_stakes=high_stakes)
        sampling = self._router.sampling("critic")
        thinking = self._router.thinking_enabled("critic")

        critic_prompt = await self._get_critic_prompt()
        messages = [
            {"role": "system", "content": critic_prompt},
            {
                "role": "user",
                "content": (
                    f"Step task: {step.input}\n"
                    f"Expected output: {step.expected_output}\n"
                    f"Actual output: {result.output[:2000]}"
                ),
            },
        ]

        try:
            resp: ChatResponse = await self._client.chat(
                model, messages, sampling=sampling, thinking=thinking
            )
            return _parse_critic_score(resp.content)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Critic parse failed: %s — defaulting to pass", exc)
            return CriticScore(
                scores={"completeness": 7, "accuracy": 7, "format": 7, "coherence": 7},
                final_score=7.0,
                retry=False,
                reason="Critic parse error, defaulting to pass",
            )
