"""Main orchestrator — wires Triage, Memory, Planner, Executor, Critic, and State."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent.config import get_settings, Settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter
from agent.core.triage import Triage
from agent.core.planner import Planner
from agent.core.executor import Executor
from agent.core.critic import Critic
from agent.infra.state import StateManager
from agent.memory.neo4j_client import MemoryClient
from agent.memory.embedder import Embedder
from agent.memory.entity_extractor import EntityExtractor
from agent.intelligence.skill_injector import get_skill_context
from agent.intelligence.lesson_injector import get_lesson_context
from agent.memory.lesson_extractor import LessonExtractor
from agent.intelligence.action_applier import ActionApplier
from agent.intelligence.prompt_manager import PromptManager
from agent.core.approval import ApprovalGate
from agent.personality.loader import load_personality, Personality
from agent.schemas import AgentResult, ScoredResult, CriticScore, StepResult, Plan

# Type alias for progress callbacks: (phase, detail) -> None or Awaitable[None]
ProgressCallback = Any  # Callable[[str, str], Awaitable[None] | None] | None

logger = logging.getLogger(__name__)


class Agent:
    """Triage → Memory Read → Plan → Execute → Critique → Memory Write."""

    def __init__(
        self,
        settings: Settings | None = None,
        approval_gate: ApprovalGate | None = None,
    ):
        self._settings = settings or get_settings()
        self._client = OllamaClient(base_url=self._settings.models.ollama_base_url)
        router = ModelRouter(self._settings)

        self._personality = load_personality()
        self._memory = MemoryClient()

        # Evolution subsystems
        self._action_applier = ActionApplier(self._settings, memory=self._memory)
        self._prompt_manager = PromptManager(memory=self._memory)
        self._episode_count = 0

        self._triage = Triage(client=self._client, router=router)
        self._planner = Planner(client=self._client, router=router, personality=self._personality, prompt_manager=self._prompt_manager)
        self._executor = Executor(
            client=self._client, router=router, settings=self._settings,
            personality=self._personality, approval_gate=approval_gate,
        )
        self._critic = Critic(client=self._client, router=router, executor=self._executor, prompt_manager=self._prompt_manager)
        self._state = StateManager(self._settings.recovery.state_file)
        self._embedder = Embedder()
        self._entity_extractor = EntityExtractor(client=self._client, router=router)
        self._lesson_extractor = LessonExtractor(client=self._client, router=router)
        self._last_episode_id: str | None = None
        self._cached_input_entities: list[dict[str, str]] | None = None

    @property
    def last_episode_id(self) -> str | None:
        """Last persisted episode ID (for feedback /good /bad)."""
        return self._last_episode_id

    @property
    def memory(self) -> MemoryClient:
        """Expose memory client for feedback and external queries."""
        return self._memory

    @property
    def action_applier(self) -> ActionApplier:
        """Expose action applier for reflection integration."""
        return self._action_applier

    @property
    def prompt_manager(self) -> PromptManager:
        """Expose prompt manager for external prompt mutations."""
        return self._prompt_manager

    async def initialize(self) -> None:
        """Connect to optional services (Neo4j). Safe to skip."""
        if self._settings.memory.entity_extraction:
            connected = await self._memory.connect()
            if connected:
                await self._memory.setup_schema()

    async def shutdown(self) -> None:
        """Clean up connections."""
        await self._client.close()
        await self._memory.close()

    async def _notify(self, cb: ProgressCallback, phase: str, detail: str) -> None:
        """Fire progress callback if set."""
        if cb is not None:
            ret = cb(phase, detail)
            if ret is not None:
                await ret

    async def run(
        self,
        user_input: str,
        on_progress: ProgressCallback = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        """Execute the full agent pipeline on user input.

        Args:
            conversation_history: list of {"role": "user"|"assistant", "content": "..."}
                from previous turns in this conversation session.
        """
        self._conversation_history = conversation_history or []
        start = time.monotonic()
        logger.info("=== Agent started: %s", user_input[:100])

        # Check for crash recovery
        resume_after = 0
        saved = self._state.load_state()
        if saved is not None:
            plan, resume_after, saved_input = saved
            if saved_input and saved_input != user_input:
                logger.warning("Input changed since saved plan — discarding old state")
                self._state.clear()
            else:
                logger.info("Resuming previous plan from step %d", resume_after + 1)
                return await self._execute_pipeline(
                    user_input, plan, resume_after, start, on_progress, phase_timings
                )

        # Phase timings for self-diagnostics
        phase_timings: dict[str, float] = {}

        # Phase 0: Triage
        logger.info("Phase 0: Triage...")
        await self._notify(on_progress, "triage", "Classifying input...")
        t0 = time.monotonic()
        complexity = await self._triage.classify(user_input)
        phase_timings["triage"] = time.monotonic() - t0
        logger.info("Triage: %s", complexity)

        if complexity == "simple":
            await self._notify(on_progress, "executing", "Direct answer...")
            # Inject active goals into history so simple path is aware of them
            simple_history = list(self._conversation_history) if self._conversation_history else []
            if self._memory.available:
                active_goals = await self._memory.get_active_goals()
                if active_goals:
                    lines = ["[Your active goals set by the user:]"]
                    for g in active_goals:
                        lines.append(f"- {g['description']}")
                    simple_history.insert(0, {"role": "system", "content": "\n".join(lines)})
            result = await self._executor.execute_direct(
                user_input, conversation_history=simple_history
            )
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
            phase_timings["execute"] = time.monotonic() - t0 - phase_timings["triage"]
            self._write_trace(user_input, agent_result, elapsed, simple=True, phase_timings=phase_timings)
            # Extract lessons even on simple path (corrections are often simple messages)
            if self._memory.available and self._settings.lessons.enabled:
                await self._extract_lessons(user_input)
            return agent_result

        # Phase 0.5: Memory read + skill injection
        self._cached_input_entities = None
        context_parts: list[str] = []
        t_mem = time.monotonic()
        if self._memory.available:
            logger.info("Phase 0.5: Memory read...")
            await self._notify(on_progress, "memory", "Loading context...")
            # Run entity extraction and embedding in parallel
            entities_task = self._entity_extractor.extract(user_input)
            embed_task = self._embedder.embed(user_input)
            entities, query_embedding = await asyncio.gather(entities_task, embed_task)
            self._cached_input_entities = entities
            entity_names = [e["name"] for e in entities]
            episode_ctx = await self._memory.get_context(
                entity_names,
                query_embedding=query_embedding or None,
            )
            if episode_ctx:
                context_parts.append(episode_ctx)
                logger.info("Memory: injecting episode context (%d chars)", len(episode_ctx))
            skill_ctx = await get_skill_context(self._memory)
            if skill_ctx:
                context_parts.append(skill_ctx)
                logger.info("Memory: injecting %s", skill_ctx.split("\n")[0])
            # Inject active goals so the agent is aware of persistent objectives
            active_goals = await self._memory.get_active_goals()
            if active_goals:
                goal_lines = ["Active persistent goals (created by the user):"]
                for g in active_goals:
                    goal_lines.append(f"  - [{g.get('priority', 3)}] {g['description']}")
                context_parts.append("\n".join(goal_lines))
                logger.info("Memory: injecting %d active goals", len(active_goals))
            # Inject lessons from past corrections
            if self._settings.lessons.enabled and query_embedding:
                lesson_ctx = await get_lesson_context(
                    self._memory,
                    query_embedding=query_embedding,
                    limit=self._settings.lessons.max_in_context,
                )
                if lesson_ctx:
                    context_parts.insert(0, lesson_ctx)
                    logger.info("Memory: injecting lesson context (%d chars)", len(lesson_ctx))
        # Inject conversation history into context
        if self._conversation_history:
            history_lines = ["Recent conversation:"]
            for msg in self._conversation_history[-10:]:  # Last 10 turns max
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:300]
                history_lines.append(f"  {role}: {content}")
            context_parts.insert(0, "\n".join(history_lines))

        context = "\n\n".join(context_parts)

        phase_timings["memory"] = time.monotonic() - t_mem

        # Phase 1: Plan
        logger.info("Phase 1: Planning...")
        await self._notify(on_progress, "planning", "Building plan...")
        t_plan = time.monotonic()
        plan = await self._planner.plan(user_input, context=context)
        phase_timings["planning"] = time.monotonic() - t_plan
        logger.info("Plan: %s (%d steps)", plan.goal, len(plan.steps))
        await self._notify(on_progress, "planning", f"Plan: {len(plan.steps)} steps")

        return await self._execute_pipeline(user_input, plan, 0, start, on_progress, phase_timings)

    async def _execute_pipeline(
        self,
        user_input: str,
        plan: object,
        resume_after: int,
        start: float,
        on_progress: ProgressCallback = None,
        phase_timings: dict[str, float] | None = None,
    ) -> AgentResult:
        """Execute the plan→execute→critique pipeline."""
        from agent.schemas import Plan
        assert isinstance(plan, Plan)

        # Save plan for crash recovery
        if self._settings.recovery.persist_cursor:
            self._state.save_plan(plan, user_input=user_input)

        if phase_timings is None:
            phase_timings = {}

        # Phase 2: Execute
        logger.info("Phase 2: Executing...")
        t_exec = time.monotonic()
        total_steps = len(plan.steps)
        await self._notify(on_progress, "executing", f"Step 0/{total_steps}")

        def _on_step_done(step_id: int) -> None:
            if self._settings.recovery.persist_cursor:
                self._state.advance_cursor(step_id)
            if on_progress is not None:
                asyncio.ensure_future(
                    self._notify(on_progress, "executing", f"Step {step_id}/{total_steps}")
                )

        results = await self._executor.execute(
            plan, on_step_done=_on_step_done, resume_after=resume_after
        )

        phase_timings["execute"] = time.monotonic() - t_exec

        # Phase 3: Critique
        logger.info("Phase 3: Critiquing...")
        t_critic = time.monotonic()
        await self._notify(on_progress, "critiquing", "Evaluating quality...")
        scored = await self._critic.evaluate(plan, results)
        phase_timings["critique"] = time.monotonic() - t_critic

        # Clear state on success
        self._state.clear()

        elapsed = time.monotonic() - start
        logger.info("=== Agent done in %.1fs", elapsed)

        agent_result = AgentResult(goal=plan.goal, results=scored)
        self._write_trace(user_input, agent_result, elapsed, phase_timings=phase_timings)

        # Phase 4: Memory write — persist episode + extract skills + extract lessons
        if self._memory.available:
            await self._persist_memory(user_input, agent_result)
            await self._extract_skills(plan, scored)
            if self._settings.lessons.enabled:
                await self._extract_lessons(user_input)

        # Phase 5: Evolution — track prompt scores + periodic strategy analysis
        self._episode_count += 1
        await self._evolution_cycle(agent_result)

        return agent_result

    async def _evolution_cycle(self, result: AgentResult) -> None:
        """Run evolution tasks: prompt scoring, strategy analysis."""
        if not self._settings.evolution.enabled:
            return

        # Record scores for prompt A/B testing
        avg_score = (
            sum(r.score.final_score for r in result.results) / len(result.results)
            if result.results else 0.0
        )
        await asyncio.gather(
            self._prompt_manager.record_score("planner", avg_score),
            self._prompt_manager.record_score("critic", avg_score),
        )
        await asyncio.gather(
            self._prompt_manager.evaluate_and_promote("planner"),
            self._prompt_manager.evaluate_and_promote("critic"),
        )

        # Run strategy evolution + workspace evolution every 20 episodes
        if self._episode_count % 20 == 0:
            try:
                from agent.intelligence.strategy_evolver import evolve_strategy
                suggestions = await evolve_strategy(self._settings.logging.trace_file)
                if suggestions:
                    await self._action_applier.apply(suggestions)
            except Exception as exc:
                logger.warning("Strategy evolution failed: %s", exc)

            await self._evolve_workspace()

    async def _evolve_workspace(self) -> list[str]:
        """Synthesize lessons into workspace personality files."""
        if not self._memory.available:
            return []
        try:
            from agent.intelligence.workspace_evolver import evolve_workspace
            from agent.models.router import ModelRouter
            router = ModelRouter(self._settings)
            modified = await evolve_workspace(
                client=self._client,
                router=router,
                memory=self._memory,
            )
            if modified:
                from agent.personality.loader import load_personality
                self._personality = load_personality()
                logger.info("Workspace evolved: %s — personality reloaded", modified)
            return modified
        except Exception as exc:
            logger.warning("Workspace evolution failed: %s", exc)
            return []

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

            # Extract entities: reuse cached input entities, only extract from outputs
            output_text = "\n".join(r.result.output[:500] for r in result.results)
            output_entities = await self._entity_extractor.extract(output_text)

            if self._cached_input_entities is not None:
                # Merge cached input entities + new output entities (dedup by name)
                seen: dict[str, dict[str, str]] = {}
                for e in self._cached_input_entities:
                    seen[e["name"]] = e
                for e in output_entities:
                    seen[e["name"]] = e  # output version wins on duplicate
                entities = list(seen.values())
            else:
                # Fallback: extract from everything (memory was unavailable during read)
                all_text = user_input + "\n" + output_text
                entities = await self._entity_extractor.extract(all_text)

            summary = f"Goal: {result.goal}. "
            summary += f"{len(result.results)} steps, avg score {avg_score:.1f}."

            # Generate embedding for the episode summary
            embedding = await self._embedder.embed(summary)

            episode_id = await self._memory.persist_episode(
                goal=result.goal,
                summary=summary,
                score=avg_score,
                entities=entities,
                previous_episode_id=self._last_episode_id,
                embedding=embedding or None,
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
        # Build a readable name from unique tools in order
        seen: set[str] = set()
        unique_tools: list[str] = []
        for t in tool_chain:
            if t not in seen:
                seen.add(t)
                unique_tools.append(t)
        skill_name = "_".join(unique_tools)
        try:
            await self._memory.persist_skill(skill_name, tool_chain, avg)
            logger.info("Skill extracted: %s (chain=%s, score=%.1f)",
                        skill_name, tool_chain, avg)
        except Exception as exc:
            logger.warning("Skill extraction failed: %s", exc)

    async def _extract_lessons(self, user_input: str) -> None:
        """Detect and persist lessons from user corrections in conversation."""
        if not self._conversation_history:
            return

        # Find the last assistant response before this user message
        last_assistant = None
        for msg in reversed(self._conversation_history):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break
        if not last_assistant:
            return

        try:
            lesson = await self._lesson_extractor.detect_and_extract(
                user_message=user_input,
                assistant_response=last_assistant,
            )
            if lesson is None:
                return

            # Embed the lesson for semantic search + dedup
            embed_text = f"{lesson['context']}: {lesson['rule']}"
            embedding = await self._embedder.embed(embed_text)

            # Dedup: check for similar existing lesson
            similar = await self._memory.find_similar_lesson(
                embedding or [],
                threshold=self._settings.lessons.similarity_threshold,
            )
            if similar:
                await self._memory.reinforce_lesson(similar["id"])
                logger.info("Lesson reinforced: %s", similar["rule"][:60])
            else:
                import uuid
                lesson_id = str(uuid.uuid4())[:16]
                await self._memory.persist_lesson(
                    lesson_id=lesson_id,
                    rule=lesson["rule"],
                    context=lesson["context"],
                    category=lesson["category"],
                    source_quote=lesson["source_quote"],
                    embedding=embedding or None,
                    episode_id=self._last_episode_id,
                )
        except Exception as exc:
            logger.warning("Lesson extraction failed: %s", exc)

    async def learn_from_feedback(self, reason: str) -> str | None:
        """Extract and persist a lesson from explicit /bad feedback reason."""
        if not self._memory.available or not self._settings.lessons.enabled:
            return None
        # Build episode summary from last episode
        episode_summary = f"Episode {self._last_episode_id or 'unknown'}"
        try:
            lesson = await self._lesson_extractor.extract_from_feedback(
                reason=reason, episode_summary=episode_summary,
            )
            if lesson is None:
                return None

            embed_text = f"{lesson['context']}: {lesson['rule']}"
            embedding = await self._embedder.embed(embed_text)

            similar = await self._memory.find_similar_lesson(
                embedding or [],
                threshold=self._settings.lessons.similarity_threshold,
            )
            if similar:
                await self._memory.reinforce_lesson(similar["id"])
                return similar["rule"]
            else:
                import uuid
                lesson_id = str(uuid.uuid4())[:16]
                await self._memory.persist_lesson(
                    lesson_id=lesson_id,
                    rule=lesson["rule"],
                    context=lesson["context"],
                    category=lesson["category"],
                    source_quote=lesson["source_quote"],
                    embedding=embedding or None,
                    episode_id=self._last_episode_id,
                )
                return lesson["rule"]
        except Exception as exc:
            logger.warning("Lesson from feedback failed: %s", exc)
            return None

    def _write_trace(
        self,
        user_input: str,
        result: AgentResult,
        elapsed: float,
        simple: bool = False,
        phase_timings: dict[str, float] | None = None,
    ) -> None:
        """Append a detailed JSONL trace entry.

        Logs per-step details (tool, input, output, errors, critic breakdown)
        so the agent can read its own logs for self-correction.
        """
        trace_path = Path(self._settings.logging.trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Build detailed step records
        step_details = []
        for sr in result.results:
            detail: dict[str, Any] = {
                "step_id": sr.step.id,
                "tool": sr.step.tool,
                "input": sr.step.input[:300],
                "output": sr.result.output[:500],
                "expected": sr.step.expected_output[:200],
                "score": sr.score.final_score,
                "scores_breakdown": sr.score.scores,
                "retry": sr.score.retry,
            }
            if sr.score.reason:
                detail["reason"] = sr.score.reason
            # Detect errors in output
            if sr.result.output.startswith("ERROR"):
                detail["error"] = sr.result.output[:300]
            step_details.append(detail)

        entry: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": user_input[:500],
            "goal": result.goal,
            "steps": len(result.results),
            "scores": [r.score.final_score for r in result.results],
            "elapsed_s": round(elapsed, 2),
            "path": "simple" if simple else "complex",
            "memory": self._memory.available,
            "step_details": step_details,
        }
        if phase_timings:
            entry["timings"] = {k: round(v, 2) for k, v in phase_timings.items()}

        # Flag episodes with errors or low scores for easy filtering
        errors = [d for d in step_details if "error" in d]
        if errors:
            entry["has_errors"] = True
            entry["error_count"] = len(errors)
        low_scores = [d for d in step_details if d["score"] < 6.5]
        if low_scores:
            entry["has_low_scores"] = True

        with open(trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
