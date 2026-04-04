"""Executor — runs plan steps by dispatching to tools or the LLM.

Supports parallel execution (asyncio.gather) and Skeleton-of-Thought for long outputs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from agent.config import get_settings, Settings
from agent.models.ollama_client import OllamaClient, ChatResponse
from agent.models.router import ModelRouter
from agent.intelligence.loop_detection import LoopDetector
from agent.personality.loader import Personality
from agent.schemas import Plan, Step, StepResult

from agent.tools.registry import get_tool, discover_tools

# Auto-discover all tools (scans agent/tools/ and agent/tools/custom/)
discover_tools()

logger = logging.getLogger(__name__)


class Executor:
    """Execute plan steps — dispatch to tools or call the LLM.

    Features:
    - Parallel execution of steps listed in plan.parallel
    - Skeleton-of-Thought for long expected outputs
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
        personality: Personality | None = None,
    ):
        self._settings = settings or get_settings()
        self._client = client or OllamaClient(
            base_url=self._settings.models.ollama_base_url
        )
        self._router = router or ModelRouter(self._settings)
        self._personality = personality
        self._loop_detector = LoopDetector()

    async def execute(
        self,
        plan: Plan,
        *,
        on_step_done: Callable[[int], Awaitable[None] | None] | None = None,
        resume_after: int = 0,
    ) -> list[StepResult]:
        """Execute plan steps, with parallel support.

        Args:
            plan: The plan to execute.
            on_step_done: Optional callback after each step completes.
            resume_after: Skip steps with id <= this value (for crash recovery).

        Returns:
            List of StepResult, one per step.
        """
        self._loop_detector.reset()
        results: list[StepResult] = []
        context_trace: list[str] = []
        parallel_ids = set(plan.parallel)

        # Separate parallel and sequential steps
        steps = [s for s in plan.steps if s.id > resume_after]

        # Group consecutive parallel steps together
        i = 0
        while i < len(steps):
            step = steps[i]

            if step.id in parallel_ids:
                # Collect all consecutive parallel steps
                parallel_batch: list[Step] = []
                while i < len(steps) and steps[i].id in parallel_ids:
                    parallel_batch.append(steps[i])
                    i += 1

                # Execute in parallel
                logger.info(
                    "Executing %d steps in parallel: %s",
                    len(parallel_batch),
                    [s.id for s in parallel_batch],
                )
                batch_results = await asyncio.gather(
                    *(self.execute_step(s, context_trace) for s in parallel_batch),
                    return_exceptions=True,
                )

                for s, r in zip(parallel_batch, batch_results):
                    if isinstance(r, Exception):
                        logger.error("Parallel step %d failed: %s", s.id, r)
                        r = StepResult(step_id=s.id, output=f"ERROR: {r}")
                    results.append(r)
                    context_trace.append(
                        f"Step {s.id} ({s.tool}): {r.output[:200]}"
                    )
                    if on_step_done is not None:
                        ret = on_step_done(s.id)
                        if ret is not None:
                            await ret
            else:
                # Sequential execution
                logger.info("Executing step %d: tool=%s", step.id, step.tool)
                result = await self.execute_step(step, context_trace)
                results.append(result)
                context_trace.append(
                    f"Step {step.id} ({step.tool}): {result.output[:200]}"
                )
                if on_step_done is not None:
                    ret = on_step_done(step.id)
                    if ret is not None:
                        await ret
                i += 1

        return results

    async def execute_step(
        self, step: Step, context_trace: list[str] | None = None
    ) -> StepResult:
        """Execute a single step."""
        if step.tool == "none":
            return await self._execute_llm(step, context_trace)

        # Loop detection
        loop_msg = self._loop_detector.check(step.tool, step.input)
        if loop_msg:
            return StepResult(step_id=step.id, output=f"LOOP DETECTED: {loop_msg}")

        tool = get_tool(step.tool)
        if tool is None:
            logger.warning("Tool '%s' not available, falling back to LLM", step.tool)
            return await self._execute_llm(step, context_trace)

        try:
            output = await tool.run(step.input)
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", step.tool, exc)
            output = f"ERROR: {exc}"

        return StepResult(step_id=step.id, output=output)

    async def execute_direct(self, user_input: str) -> StepResult:
        """Direct LLM call without a plan (for simple triage bypass)."""
        model = self._router.select("executor")
        sampling = self._router.sampling("executor")

        system_parts = [
            "You are a helpful assistant. Always write complete words "
            "with proper accents and diacritics (é, è, ê, à, ç, ù, ô, etc.). "
            "Never truncate words. Respond in the same language as the user."
        ]
        if self._personality and self._personality.soul:
            system_parts.append(f"\nPersonality:\n{self._personality.soul}")
        if self._personality and self._personality.user:
            system_parts.append(f"\nUser profile:\n{self._personality.user}")

        resp: ChatResponse = await self._client.chat(
            model,
            [
                {"role": "system", "content": "\n".join(system_parts)},
                {"role": "user", "content": user_input},
            ],
            sampling=sampling,
        )
        return StepResult(step_id=0, output=resp.content, tokens_used=resp.eval_count)

    async def execute_sot(self, step: Step) -> StepResult:
        """Skeleton-of-Thought: generate skeleton then fill sections in parallel.

        Used when expected output is long (> skeleton_threshold tokens).
        """
        model = self._router.select("executor")
        sampling = self._router.sampling("executor")
        thinking = self._router.thinking_enabled("executor")

        # Pass 1: Generate skeleton
        skeleton_resp = await self._client.chat(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Generate ONLY section titles and a one-sentence summary "
                        "for each. No content. Output as a numbered list."
                    ),
                },
                {"role": "user", "content": step.input},
            ],
            sampling=sampling,
            thinking=thinking,
        )
        skeleton = skeleton_resp.content

        # Parse sections from skeleton
        sections = self._parse_skeleton_sections(skeleton)
        if not sections:
            # Fallback to regular execution
            return await self._execute_llm(step)

        logger.info("SoT: %d sections to fill in parallel", len(sections))

        # Pass 2: Fill each section in parallel
        fill_tasks = [
            self._fill_section(model, sampling, thinking, step.input, section)
            for section in sections
        ]
        section_results = await asyncio.gather(*fill_tasks, return_exceptions=True)

        # Assemble
        parts: list[str] = []
        for section, result in zip(sections, section_results):
            if isinstance(result, Exception):
                parts.append(f"## {section}\n[Error: {result}]")
            else:
                parts.append(f"## {section}\n{result}")

        output = "\n\n".join(parts)
        return StepResult(
            step_id=step.id,
            output=output,
            tokens_used=skeleton_resp.eval_count,
        )

    async def _fill_section(
        self,
        model: str,
        sampling: object,
        thinking: bool,
        task: str,
        section: str,
    ) -> str:
        """Fill a single section from the SoT skeleton."""
        resp = await self._client.chat(
            model,
            [
                {
                    "role": "user",
                    "content": (
                        f"Context: {task}\n\n"
                        f"Write the content for this section ONLY: {section}\n"
                        f"Be concise and focused."
                    ),
                },
            ],
            sampling=sampling,
            thinking=thinking,
        )
        return resp.content

    def _parse_skeleton_sections(self, skeleton: str) -> list[str]:
        """Extract section titles from a numbered list."""
        sections: list[str] = []
        for line in skeleton.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip numbering: "1. Title" → "Title", "- Title" → "Title"
            for prefix in ("- ", "* "):
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            if line and line[0].isdigit():
                # Strip "1. " or "1) " style
                parts = line.split(". ", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    line = parts[1]
                else:
                    parts = line.split(") ", 1)
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        line = parts[1]
            # Take just the title (before any ":" summary)
            title = line.split(":")[0].strip().strip("#").strip()
            if title:
                sections.append(title)
        return sections

    async def _execute_llm(
        self, step: Step, context_trace: list[str] | None = None
    ) -> StepResult:
        """Execute a step via LLM generation."""
        # Check if SoT should be activated
        sot_threshold = self._settings.skeleton.token_threshold
        if (
            self._settings.skeleton.enabled
            and len(step.expected_output) > 50  # heuristic: long expected output
            and "comprehensive" in step.input.lower()
            or "detailed" in step.input.lower()
            or "guide" in step.input.lower()
        ):
            logger.info("Step %d: SoT activated (long output expected)", step.id)
            return await self.execute_sot(step)

        model = self._router.select("executor")
        sampling = self._router.sampling("executor")
        thinking = self._router.thinking_enabled("executor")

        messages: list[dict[str, str]] = []
        if context_trace:
            trace = "\n".join(context_trace)
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous step results:\n{trace}",
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Task: {step.input}\n"
                    f"Expected output: {step.expected_output}\n"
                    f"Produce the expected output directly."
                ),
            }
        )

        resp: ChatResponse = await self._client.chat(
            model, messages, sampling=sampling, thinking=thinking
        )
        return StepResult(
            step_id=step.id, output=resp.content, tokens_used=resp.eval_count
        )
