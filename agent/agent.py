"""Main orchestrator — wires Triage, Memory, Planner, Executor, Critic, and State."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.config import get_settings, Settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter
from agent.core.triage import Triage
from agent.core.planner import Planner
from agent.core.executor import Executor
from agent.core.critic import Critic
from agent.infra.state import StateManager
from agent.memory.neo4j_client import MemoryClient
from agent.memory.entity_extractor import EntityExtractor
from agent.intelligence.skill_injector import get_skill_context
from agent.personality.loader import load_personality, Personality
from agent.schemas import AgentResult, ScoredResult, CriticScore, StepResult, Plan

logger = logging.getLogger(__name__)


class Agent:
    """Triage → Memory Read → Plan → Execute → Critique → Memory Write."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        client = OllamaClient(base_url=self._settings.models.ollama_base_url)
        router = ModelRouter(self._settings)

        self._personality = load_personality()
        self._triage = Triage(client=client, router=router)
        self._planner = Planner(client=client, router=router, personality=self._personality)
        self._executor = Executor(client=client, router=router, settings=self._settings, personality=self._personality)
        self._critic = Critic(client=client, router=router, executor=self._executor)
        self._state = StateManager(self._settings.recovery.state_file)
        self._memory = MemoryClient()
        self._entity_extractor = EntityExtractor(client=client, router=router)
        self._last_episode_id: str | None = None

    @property
    def last_episode_id(self) -> str | None:
        """Last persisted episode ID (for feedback /good /bad)."""
        return self._last_episode_id

    @property
    def memory(self) -> MemoryClient:
        """Expose memory client for feedback and external queries."""
        return self._memory

    async def initialize(self) -> None:
        """Connect to optional services (Neo4j). Safe to skip."""
        if self._settings.memory.entity_extraction:
            connected = await self._memory.connect()
            if connected:
                await self._memory.setup_schema()

    async def shutdown(self) -> None:
        """Clean up connections."""
        await self._memory.close()

    async def run(self, user_input: str) -> AgentResult:
        """Execute the full agent pipeline on user input."""
        start = time.monotonic()
        logger.info("=== Agent started: %s", user_input[:100])

        # Check for crash recovery
        resume_after = 0
        saved = self._state.load_state()
        if saved is not None:
            plan, resume_after = saved
            logger.info("Resuming previous plan from step %d", resume_after + 1)
            return await self._execute_pipeline(
                user_input, plan, resume_after, start
            )

        # Phase 0: Triage
        logger.info("Phase 0: Triage...")
        complexity = await self._triage.classify(user_input)
        logger.info("Triage: %s", complexity)

        if complexity == "simple":
            result = await self._executor.execute_direct(user_input)
            elapsed = time.monotonic() - start
            logger.info("=== Simple path done in %.1fs", elapsed)
            score = CriticScore(
                scores={"completeness": 8, "accuracy": 8, "format": 8, "coherence": 8},
                final_score=8.0,
                retry=False,
            )
            from agent.schemas import Step
            dummy_step = Step(
                id=0, tool="none", input=user_input, expected_output="direct answer"
            )
            agent_result = AgentResult(
                goal=user_input[:100],
                results=[ScoredResult(step=dummy_step, result=result, score=score)],
            )
            self._write_trace(user_input, agent_result, elapsed, simple=True)
            return agent_result

        # Phase 0.5: Memory read + skill injection
        context_parts: list[str] = []
        if self._memory.available:
            logger.info("Phase 0.5: Memory read...")
            entities = await self._entity_extractor.extract(user_input)
            entity_names = [e["name"] for e in entities]
            episode_ctx = await self._memory.get_context(entity_names)
            if episode_ctx:
                context_parts.append(episode_ctx)
                logger.info("Memory: injecting episode context (%d chars)", len(episode_ctx))
            skill_ctx = await get_skill_context(self._memory)
            if skill_ctx:
                context_parts.append(skill_ctx)
                logger.info("Memory: injecting %s", skill_ctx.split("\n")[0])
        context = "\n\n".join(context_parts)

        # Phase 1: Plan
        logger.info("Phase 1: Planning...")
        plan = await self._planner.plan(user_input, context=context)
        logger.info("Plan: %s (%d steps)", plan.goal, len(plan.steps))

        return await self._execute_pipeline(user_input, plan, 0, start)

    async def _execute_pipeline(
        self,
        user_input: str,
        plan: object,
        resume_after: int,
        start: float,
    ) -> AgentResult:
        """Execute the plan→execute→critique pipeline."""
        from agent.schemas import Plan
        assert isinstance(plan, Plan)

        # Save plan for crash recovery
        if self._settings.recovery.persist_cursor:
            self._state.save_plan(plan)

        # Phase 2: Execute
        logger.info("Phase 2: Executing...")

        def _on_step_done(step_id: int) -> None:
            if self._settings.recovery.persist_cursor:
                self._state.advance_cursor(step_id)

        results = await self._executor.execute(
            plan, on_step_done=_on_step_done, resume_after=resume_after
        )

        # Phase 3: Critique
        logger.info("Phase 3: Critiquing...")
        scored = await self._critic.evaluate(plan, results)

        # Clear state on success
        self._state.clear()

        elapsed = time.monotonic() - start
        logger.info("=== Agent done in %.1fs", elapsed)

        agent_result = AgentResult(goal=plan.goal, results=scored)
        self._write_trace(user_input, agent_result, elapsed)

        # Phase 4: Memory write — persist episode + extract skills
        if self._memory.available:
            await self._persist_memory(user_input, agent_result)
            await self._extract_skills(plan, scored)

        return agent_result

    async def _persist_memory(
        self, user_input: str, result: AgentResult
    ) -> None:
        """Extract entities and persist the episode to Neo4j."""
        try:
            avg_score = (
                sum(r.score.final_score for r in result.results) / len(result.results)
                if result.results
                else 0.0
            )

            # Collect text for entity extraction
            all_text = user_input + "\n" + "\n".join(
                r.result.output[:500] for r in result.results
            )
            entities = await self._entity_extractor.extract(all_text)

            summary = f"Goal: {result.goal}. "
            summary += f"{len(result.results)} steps, avg score {avg_score:.1f}."

            episode_id = await self._memory.persist_episode(
                goal=result.goal,
                summary=summary,
                score=avg_score,
                entities=entities,
                previous_episode_id=self._last_episode_id,
            )
            if episode_id:
                self._last_episode_id = episode_id
                logger.info("Memory: episode %s persisted", episode_id)

        except Exception as exc:
            logger.warning("Memory write failed: %s", exc)

    async def _extract_skills(
        self, plan: Plan, scored: list[ScoredResult]
    ) -> None:
        """Extract reusable skills from high-scoring tool chains."""
        if not scored:
            return
        avg = sum(s.score.final_score for s in scored) / len(scored)
        if avg < 8.0:
            return
        tool_chain = [s.step.tool for s in scored if s.step.tool != "none"]
        if len(tool_chain) < 2:
            return
        skill_name = f"{plan.goal[:50]}_{hash(tuple(tool_chain)) % 10000}"
        try:
            await self._memory.persist_skill(skill_name, tool_chain, avg)
            logger.info("Skill extracted: %s (chain=%s, score=%.1f)",
                        skill_name, tool_chain, avg)
        except Exception as exc:
            logger.warning("Skill extraction failed: %s", exc)

    def _write_trace(
        self,
        user_input: str,
        result: AgentResult,
        elapsed: float,
        simple: bool = False,
    ) -> None:
        """Append a JSONL trace entry."""
        trace_path = Path(self._settings.logging.trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "input": user_input[:500],
            "goal": result.goal,
            "steps": len(result.results),
            "scores": [r.score.final_score for r in result.results],
            "elapsed_s": round(elapsed, 2),
            "path": "simple" if simple else "complex",
            "memory": self._memory.available,
        }
        with open(trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
