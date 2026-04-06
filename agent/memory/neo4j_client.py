"""Neo4j memory client — episodic graph memory with graceful degradation."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.config import get_settings

logger = logging.getLogger(__name__)

# Neo4j driver is an optional dependency
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    AsyncGraphDatabase = None  # type: ignore
    AsyncDriver = None  # type: ignore


class MemoryClient:
    """Read/write episodic memory in Neo4j.

    Graceful degradation: if Neo4j is unavailable or the driver is not installed,
    all operations return empty results and log warnings. The agent works fine without memory.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        settings = get_settings()
        self._uri = uri or settings.memory.neo4j_uri
        self._user = user or settings.memory.neo4j_user
        self._password = (
            password
            or os.environ.get("NEO4J_PASSWORD")
            or settings.memory.neo4j_password
        )
        self._context_k = settings.memory.context_k
        self._write_threshold = settings.memory.write_threshold
        self._read_threshold = settings.memory.read_threshold
        self._skills_cache_ttl = settings.memory.skills_cache_ttl
        self._driver: Any = None
        self._available = False
        self._skills_cache: list[dict] | None = None
        self._skills_cache_ts: float = 0.0

    async def connect(self) -> bool:
        """Attempt to connect to Neo4j. Returns True if successful."""
        if not HAS_NEO4J:
            logger.warning("Neo4j driver not installed (pip install neo4j)")
            return False

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            await self._driver.verify_connectivity()
            self._available = True
            logger.info("Neo4j connected: %s", self._uri)
            return True
        except Exception as exc:
            logger.warning("Neo4j unavailable: %s — memory disabled", exc)
            self._available = False
            return False

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def setup_schema(self) -> None:
        """Run schema constraints and indexes."""
        if not self._available:
            return

        settings = get_settings()
        dims = settings.memory.embedding_dimensions
        queries = [
            "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE INDEX episode_score IF NOT EXISTS FOR (ep:Episode) ON (ep.score)",
            "CREATE INDEX episode_timestamp IF NOT EXISTS FOR (ep:Episode) ON (ep.timestamp)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE INDEX job_status IF NOT EXISTS FOR (j:Job) ON (j.status)",
            "CREATE CONSTRAINT lesson_id IF NOT EXISTS FOR (l:Lesson) REQUIRE l.id IS UNIQUE",
            "CREATE INDEX lesson_category IF NOT EXISTS FOR (l:Lesson) ON (l.category)",
        ]
        vector_query = (
            "CREATE VECTOR INDEX episode_embedding IF NOT EXISTS "
            "FOR (ep:Episode) ON (ep.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {dims}, "
            "`vector.similarity_function`: 'cosine'"
            "}}"
        )
        try:
            async with self._driver.session() as session:
                for q in queries:
                    await session.run(q)
                try:
                    await session.run(vector_query)
                except Exception as vec_exc:
                    logger.warning("Vector index creation failed (Neo4j <5.11?): %s", vec_exc)
                lesson_vector_query = (
                    "CREATE VECTOR INDEX lesson_embedding IF NOT EXISTS "
                    "FOR (l:Lesson) ON (l.embedding) "
                    "OPTIONS {indexConfig: {"
                    f"`vector.dimensions`: {dims}, "
                    "`vector.similarity_function`: 'cosine'"
                    "}}"
                )
                try:
                    await session.run(lesson_vector_query)
                except Exception as vec_exc:
                    logger.warning("Lesson vector index creation failed: %s", vec_exc)
            logger.info("Neo4j schema setup complete")
        except Exception as exc:
            logger.warning("Neo4j schema setup failed: %s", exc)

    async def get_context(
        self,
        detected_entities: list[str],
        query_embedding: list[float] | None = None,
    ) -> str:
        """Query top-K relevant episodes via hybrid search (vector + lexical).

        Uses vector similarity (weighted by config vector_weight) combined with
        lexical entity matching (weighted by config lexical_weight).
        Falls back to lexical-only if no embedding is provided.

        Returns a formatted context string for injection into the planner prompt.
        Returns empty string if Neo4j is unavailable.
        """
        if not self._available:
            return ""
        if not detected_entities and not query_embedding:
            return ""

        settings = get_settings()
        vector_w = settings.memory.vector_weight
        lexical_w = settings.memory.lexical_weight

        # {episode_id: {"summary": ..., "ts": ..., "goal": ..., "score": float}}
        results: dict[str, dict] = {}

        try:
            async with self._driver.session() as session:
                # Branch 1: lexical entity match
                if detected_entities:
                    lexical_result = await session.run(
                        """
                        MATCH (e:Entity)<-[:HAS_ENTITY]-(ep:Episode)
                        WHERE e.name IN $entities AND ep.score > $threshold
                        RETURN ep.id AS id, ep.summary AS summary,
                               ep.timestamp AS ts, ep.goal AS goal
                        ORDER BY ep.timestamp DESC
                        LIMIT $limit
                        """,
                        entities=detected_entities,
                        threshold=self._read_threshold,
                        limit=self._context_k,
                    )
                    async for r in lexical_result:
                        results[r["id"]] = {
                            "summary": r["summary"],
                            "ts": r["ts"],
                            "goal": r["goal"],
                            "score": lexical_w,
                        }

                # Branch 2: vector similarity
                if query_embedding:
                    try:
                        vector_result = await session.run(
                            """
                            CALL db.index.vector.queryNodes(
                                'episode_embedding', $k, $embedding
                            ) YIELD node, score
                            WHERE node.score > $threshold
                            RETURN node.id AS id, node.summary AS summary,
                                   node.timestamp AS ts, node.goal AS goal,
                                   score AS similarity
                            """,
                            k=self._context_k,
                            embedding=query_embedding,
                            threshold=self._read_threshold,
                        )
                        async for r in vector_result:
                            ep_id = r["id"]
                            if ep_id in results:
                                results[ep_id]["score"] += vector_w * r["similarity"]
                            else:
                                results[ep_id] = {
                                    "summary": r["summary"],
                                    "ts": r["ts"],
                                    "goal": r["goal"],
                                    "score": vector_w * r["similarity"],
                                }
                    except Exception as vec_exc:
                        logger.debug("Vector search unavailable: %s", vec_exc)

            if not results:
                return ""

            # Rank by hybrid score, take top-K
            ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
            top = ranked[: self._context_k]

            lines = []
            for r in top:
                ts = r["ts"] or "unknown"
                lines.append(f"- [{ts}] {r['goal']}: {r['summary']}")

            context = "Relevant past episodes:\n" + "\n".join(lines)
            logger.info("Memory: enriched context with %d episodes (hybrid)", len(top))
            return context

        except Exception as exc:
            logger.warning("Memory read failed: %s", exc)
            return ""

    async def persist_episode(
        self,
        goal: str,
        summary: str,
        score: float,
        entities: list[dict[str, str]],
        previous_episode_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> str | None:
        """Persist an episode with its entities to the graph.

        Args:
            goal: The episode's goal.
            summary: Summary of what happened.
            score: Average critic score.
            entities: List of {"name": ..., "type": ..., "description": ...}.
            previous_episode_id: ID of the previous episode to chain with.
            embedding: Optional vector embedding of the episode summary.

        Returns:
            The episode ID, or None if persistence failed.
        """
        if not self._available:
            return None

        if score < self._write_threshold:
            logger.debug("Score %.1f below threshold %.1f, skipping persist", score, self._write_threshold)
            return None

        episode_id = str(uuid.uuid4())[:16]

        try:
            async with self._driver.session() as session:
                tx = await session.begin_transaction()
                try:
                    # Create episode
                    create_params: dict[str, Any] = {
                        "id": episode_id,
                        "summary": summary,
                        "score": score,
                        "goal": goal,
                    }
                    set_clause = (
                        "SET ep.timestamp = datetime(), "
                        "ep.summary = $summary, ep.score = $score, ep.goal = $goal"
                    )
                    if embedding is not None:
                        set_clause += ", ep.embedding = $embedding"
                        create_params["embedding"] = embedding

                    await tx.run(
                        f"MERGE (ep:Episode {{id: $id}}) {set_clause}",
                        **create_params,
                    )

                    # Batch link entities via UNWIND
                    if entities:
                        await tx.run(
                            """
                            UNWIND $entities AS ent
                            MERGE (e:Entity {name: ent.name})
                            SET e.type = ent.type, e.description = ent.description
                            WITH e
                            MATCH (ep:Episode {id: $episode_id})
                            MERGE (ep)-[:HAS_ENTITY]->(e)
                            """,
                            entities=[
                                {
                                    "name": e.get("name", ""),
                                    "type": e.get("type", ""),
                                    "description": e.get("description", ""),
                                }
                                for e in entities
                            ],
                            episode_id=episode_id,
                        )

                    # Chain with previous episode
                    if previous_episode_id:
                        await tx.run(
                            """
                            MATCH (ep:Episode {id: $id})
                            MATCH (prev:Episode {id: $prev_id})
                            MERGE (ep)-[:FOLLOWS]->(prev)
                            """,
                            id=episode_id,
                            prev_id=previous_episode_id,
                        )

                    await tx.commit()
                except Exception:
                    await tx.rollback()
                    raise

            logger.info(
                "Memory: persisted episode %s (score=%.1f, %d entities)",
                episode_id, score, len(entities),
            )
            return episode_id

        except Exception as exc:
            logger.warning("Memory write failed: %s", exc)
            return None

    async def update_episode_score(
        self, episode_id: str, score: float, reason: str | None = None
    ) -> None:
        """Override an episode's score (user feedback /good or /bad)."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                query = "MATCH (ep:Episode {id: $id}) SET ep.score = $score"
                params: dict = {"id": episode_id, "score": score}
                if reason:
                    query += ", ep.feedback_reason = $reason"
                    params["reason"] = reason
                await session.run(query, **params)
            logger.info("Memory: updated episode %s score to %.1f", episode_id, score)
        except Exception as exc:
            logger.warning("Memory score update failed: %s", exc)

    async def persist_skill(
        self, name: str, tool_chain: list[str], score: float
    ) -> None:
        """Create or update a Skill node from a successful tool chain."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (s:Skill {name: $name})
                    SET s.tool_chain = $tool_chain,
                        s.success_rate = CASE WHEN s.usage_count IS NULL
                          THEN $score
                          ELSE (s.success_rate * s.usage_count + $score) / (s.usage_count + 1)
                        END,
                        s.usage_count = coalesce(s.usage_count, 0) + 1,
                        s.last_used = datetime()
                    """,
                    name=name, tool_chain=tool_chain, score=score,
                )
            self._skills_cache = None  # invalidate cache
            logger.info("Memory: persisted skill '%s' (score=%.1f)", name, score)
        except Exception as exc:
            logger.warning("Memory skill write failed: %s", exc)

    async def get_skills(self, min_score: float = 7.0, limit: int = 5) -> list[dict]:
        """Retrieve top skills for injection into the Planner (cached with TTL)."""
        if not self._available:
            return []

        now = time.monotonic()
        if self._skills_cache is not None and (now - self._skills_cache_ts) < self._skills_cache_ttl:
            return self._skills_cache

        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (s:Skill) WHERE s.success_rate > $min_score
                    RETURN s.name AS name, s.tool_chain AS tool_chain,
                           s.success_rate AS score, s.usage_count AS uses
                    ORDER BY s.success_rate DESC LIMIT $limit
                    """,
                    min_score=min_score, limit=limit,
                )
                skills = [dict(r) async for r in result]
            self._skills_cache = skills
            self._skills_cache_ts = now
            return skills
        except Exception as exc:
            logger.warning("Memory skill read failed: %s", exc)
            return []

    # ── Lessons (user corrections / preferences) ──────────────────────

    async def persist_lesson(
        self,
        lesson_id: str,
        rule: str,
        context: str,
        category: str,
        source_quote: str,
        embedding: list[float] | None = None,
        episode_id: str | None = None,
    ) -> str | None:
        """Create a new Lesson node in the graph."""
        if not self._available:
            return None
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    CREATE (l:Lesson {
                        id: $id, rule: $rule, context: $context,
                        category: $category, source_quote: $source_quote,
                        confidence: 1.0, times_reinforced: 0,
                        created_at: datetime(), last_used: datetime(),
                        embedding: $embedding
                    })
                    """,
                    id=lesson_id, rule=rule, context=context,
                    category=category, source_quote=source_quote,
                    embedding=embedding or [],
                )
                if episode_id:
                    await session.run(
                        """
                        MATCH (l:Lesson {id: $lid}), (ep:Episode {id: $eid})
                        MERGE (l)-[:LEARNED_FROM]->(ep)
                        """,
                        lid=lesson_id, eid=episode_id,
                    )
            logger.info("Memory: persisted lesson %s — %s", lesson_id, rule[:60])
            return lesson_id
        except Exception as exc:
            logger.warning("Lesson persist failed: %s", exc)
            return None

    async def find_similar_lesson(
        self,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> dict | None:
        """Find the most similar existing lesson by vector search.

        Returns {"id", "rule", "similarity"} if above threshold, else None.
        """
        if not self._available or not embedding:
            return None
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'lesson_embedding', 1, $embedding
                    ) YIELD node, score
                    WHERE score > $threshold
                    RETURN node.id AS id, node.rule AS rule, score AS similarity
                    LIMIT 1
                    """,
                    embedding=embedding, threshold=threshold,
                )
                record = await result.single()
                if record:
                    return dict(record)
            return None
        except Exception as exc:
            logger.debug("Lesson similarity search failed: %s", exc)
            return None

    async def reinforce_lesson(self, lesson_id: str) -> None:
        """Reinforce an existing lesson — bump confidence and usage count."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MATCH (l:Lesson {id: $id})
                    SET l.times_reinforced = l.times_reinforced + 1,
                        l.confidence = l.confidence + 0.1,
                        l.last_used = datetime()
                    """,
                    id=lesson_id,
                )
            logger.info("Memory: reinforced lesson %s", lesson_id)
        except Exception as exc:
            logger.warning("Lesson reinforce failed: %s", exc)

    async def get_relevant_lessons(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve semantically relevant lessons via vector search."""
        if not self._available or not query_embedding:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'lesson_embedding', $limit, $embedding
                    ) YIELD node, score
                    RETURN node.id AS id, node.rule AS rule,
                           node.context AS context, node.category AS category,
                           node.confidence AS confidence,
                           node.times_reinforced AS times_reinforced,
                           score AS similarity
                    ORDER BY score * node.confidence DESC
                    LIMIT $limit
                    """,
                    embedding=query_embedding, limit=limit,
                )
                return [dict(r) async for r in result]
        except Exception as exc:
            logger.warning("Lesson retrieval failed: %s", exc)
            return []

    async def persist_goal(
        self,
        goal_id: str,
        description: str,
        priority: int = 3,
        parent_goal_id: str | None = None,
    ) -> None:
        """Create or update a persistent Goal node."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (g:Goal {id: $id})
                    SET g.description = $description,
                        g.status = coalesce(g.status, "active"),
                        g.priority = $priority,
                        g.created_at = coalesce(g.created_at, datetime())
                    """,
                    id=goal_id, description=description, priority=priority,
                )
                if parent_goal_id:
                    await session.run(
                        """
                        MATCH (parent:Goal {id: $parent_id})
                        MATCH (child:Goal {id: $child_id})
                        MERGE (parent)-[:DECOMPOSED_INTO]->(child)
                        """,
                        parent_id=parent_goal_id, child_id=goal_id,
                    )
            logger.info("Memory: persisted goal '%s'", goal_id)
        except Exception as exc:
            logger.warning("Memory goal write failed: %s", exc)

    async def get_active_goals(self) -> list[dict]:
        """Retrieve active goals ordered by priority."""
        if not self._available:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (g:Goal {status: "active"})
                    RETURN g.id AS id, g.description AS description,
                           g.priority AS priority
                    ORDER BY g.priority ASC, g.created_at ASC
                    """,
                )
                return [dict(r) async for r in result]
        except Exception as exc:
            logger.warning("Memory goal read failed: %s", exc)
            return []

    async def complete_goal(self, goal_id: str, episode_id: str | None = None) -> None:
        """Mark a goal as done, optionally linking to the achieving episode."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    'MATCH (g:Goal {id: $id}) SET g.status = "done"',
                    id=goal_id,
                )
                if episode_id:
                    await session.run(
                        """
                        MATCH (g:Goal {id: $goal_id})
                        MATCH (ep:Episode {id: $ep_id})
                        MERGE (g)-[:ACHIEVED_BY]->(ep)
                        """,
                        goal_id=goal_id, ep_id=episode_id,
                    )
        except Exception as exc:
            logger.warning("Memory goal complete failed: %s", exc)

    async def compress_old_episodes(self) -> int:
        """Remove old low-scoring episodes to keep the graph compact.

        Deletes episodes older than compress_after_days with score below
        read_threshold, preserving at least compress_min_keep recent episodes.

        Returns the number of deleted episodes.
        """
        if not self._available:
            return 0

        settings = get_settings()
        max_age_days = settings.memory.compress_after_days
        min_keep = settings.memory.compress_min_keep

        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (ep:Episode)
                    WHERE ep.score < $threshold
                      AND ep.timestamp < datetime() - duration({days: $max_age_days})
                    WITH ep ORDER BY ep.timestamp DESC
                    SKIP $min_keep
                    WITH collect(ep) AS to_delete
                    UNWIND to_delete AS ep
                    DETACH DELETE ep
                    RETURN count(*) AS deleted
                    """,
                    threshold=self._read_threshold,
                    max_age_days=max_age_days,
                    min_keep=min_keep,
                )
                record = await result.single()
                deleted = record["deleted"] if record else 0
                if deleted:
                    logger.info("Memory: compressed %d old episodes", deleted)
                return deleted
        except Exception as exc:
            logger.warning("Memory compression failed: %s", exc)
            return 0

    # -- Job persistence -------------------------------------------------------

    async def persist_job(
        self,
        job_id: str,
        user_input: str,
        status: str,
        chat_id: int,
        current_phase: str = "",
        progress: str = "",
        created_at: str = "",
        started_at: str | None = None,
        completed_at: str | None = None,
        error: str | None = None,
    ) -> None:
        """Create or update a Job node."""
        if not self._available:
            return

        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (j:Job {id: $id})
                    SET j.user_input = $user_input,
                        j.status = $status,
                        j.chat_id = $chat_id,
                        j.current_phase = $current_phase,
                        j.progress = $progress,
                        j.created_at = $created_at,
                        j.started_at = $started_at,
                        j.completed_at = $completed_at,
                        j.error = $error
                    """,
                    id=job_id,
                    user_input=user_input[:500],
                    status=status,
                    chat_id=chat_id,
                    current_phase=current_phase,
                    progress=progress,
                    created_at=created_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    error=error,
                )
        except Exception as exc:
            logger.warning("Job persist failed: %s", exc)

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        current_phase: str = "",
        progress: str = "",
        completed_at: str | None = None,
        error: str | None = None,
    ) -> None:
        """Lightweight job status update."""
        if not self._available:
            return

        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MATCH (j:Job {id: $id})
                    SET j.status = $status,
                        j.current_phase = $current_phase,
                        j.progress = $progress,
                        j.completed_at = $completed_at,
                        j.error = $error
                    """,
                    id=job_id,
                    status=status,
                    current_phase=current_phase,
                    progress=progress,
                    completed_at=completed_at,
                    error=error,
                )
        except Exception as exc:
            logger.warning("Job status update failed: %s", exc)

    async def get_active_jobs(self, chat_id: int | None = None) -> list[dict]:
        """Retrieve pending/running jobs, optionally filtered by chat_id."""
        if not self._available:
            return []

        try:
            async with self._driver.session() as session:
                if chat_id is not None:
                    result = await session.run(
                        """
                        MATCH (j:Job) WHERE j.status IN ["pending", "running"]
                          AND j.chat_id = $chat_id
                        RETURN j ORDER BY j.created_at DESC LIMIT 20
                        """,
                        chat_id=chat_id,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (j:Job) WHERE j.status IN ["pending", "running"]
                        RETURN j ORDER BY j.created_at DESC LIMIT 20
                        """,
                    )
                records = await result.data()
                return [dict(r["j"]) for r in records]
        except Exception as exc:
            logger.warning("Get active jobs failed: %s", exc)
            return []

    async def link_job_episode(self, job_id: str, episode_id: str) -> None:
        """Link a completed Job to its resulting Episode."""
        if not self._available:
            return

        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MATCH (j:Job {id: $job_id}), (ep:Episode {id: $episode_id})
                    MERGE (j)-[:PRODUCED]->(ep)
                    """,
                    job_id=job_id,
                    episode_id=episode_id,
                )
        except Exception as exc:
            logger.warning("Link job-episode failed: %s", exc)

    # ------------------------------------------------------------------
    # Generated Tools (evolution)
    # ------------------------------------------------------------------

    async def persist_generated_tool(
        self,
        name: str,
        version: int,
        source_hash: str,
    ) -> None:
        """Create or update a GeneratedTool node."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (t:GeneratedTool {name: $name})
                    SET t.version = $version,
                        t.source_hash = $source_hash,
                        t.updated_at = datetime(),
                        t.usage_count = COALESCE(t.usage_count, 0),
                        t.error_count = COALESCE(t.error_count, 0),
                        t.avg_score = COALESCE(t.avg_score, 0.0)
                    ON CREATE SET t.created_at = datetime()
                    """,
                    name=name,
                    version=version,
                    source_hash=source_hash,
                )
        except Exception as exc:
            logger.warning("Persist generated tool failed: %s", exc)

    async def update_tool_stats(
        self,
        name: str,
        success: bool = True,
        score: float = 0.0,
    ) -> None:
        """Increment usage/error counts and update rolling avg score."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                error_inc = 0 if success else 1
                await session.run(
                    """
                    MATCH (t:GeneratedTool {name: $name})
                    SET t.usage_count = t.usage_count + 1,
                        t.error_count = t.error_count + $error_inc,
                        t.avg_score = CASE
                            WHEN t.usage_count = 0 THEN $score
                            ELSE (t.avg_score * t.usage_count + $score) / (t.usage_count + 1)
                        END,
                        t.last_used = datetime()
                    """,
                    name=name,
                    error_inc=error_inc,
                    score=score,
                )
        except Exception as exc:
            logger.warning("Update tool stats failed: %s", exc)

    async def get_tool_stats(self, name: str) -> dict | None:
        """Get stats for a single generated tool."""
        if not self._available:
            return None
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    "MATCH (t:GeneratedTool {name: $name}) RETURN t",
                    name=name,
                )
                records = await result.data()
                return dict(records[0]["t"]) if records else None
        except Exception as exc:
            logger.warning("Get tool stats failed: %s", exc)
            return None

    async def get_generated_tools(self) -> list[dict]:
        """Get all generated tool stats."""
        if not self._available:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    "MATCH (t:GeneratedTool) RETURN t ORDER BY t.usage_count DESC"
                )
                records = await result.data()
                return [dict(r["t"]) for r in records]
        except Exception as exc:
            logger.warning("Get generated tools failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt Versions (evolution)
    # ------------------------------------------------------------------

    async def persist_prompt_version(
        self,
        prompt_id: str,
        role: str,
        version: int,
        content: str,
        is_active: bool = False,
    ) -> None:
        """Create a PromptVersion node."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (p:PromptVersion {id: $id})
                    SET p.role = $role,
                        p.version = $version,
                        p.content = $content,
                        p.is_active = $is_active,
                        p.score_avg = 0.0,
                        p.usage_count = 0,
                        p.created_at = datetime()
                    """,
                    id=prompt_id,
                    role=role,
                    version=version,
                    content=content,
                    is_active=is_active,
                )
        except Exception as exc:
            logger.warning("Persist prompt version failed: %s", exc)

    async def get_active_prompt(self, role: str) -> dict | None:
        """Get the currently active prompt for a role."""
        if not self._available:
            return None
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (p:PromptVersion {role: $role, is_active: true})
                    RETURN p ORDER BY p.version DESC LIMIT 1
                    """,
                    role=role,
                )
                records = await result.data()
                return dict(records[0]["p"]) if records else None
        except Exception as exc:
            logger.warning("Get active prompt failed: %s", exc)
            return None

    async def get_candidate_prompt(self, role: str) -> dict | None:
        """Get the candidate (non-active, latest) prompt for A/B testing."""
        if not self._available:
            return None
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (p:PromptVersion {role: $role, is_active: false})
                    RETURN p ORDER BY p.version DESC LIMIT 1
                    """,
                    role=role,
                )
                records = await result.data()
                return dict(records[0]["p"]) if records else None
        except Exception as exc:
            logger.warning("Get candidate prompt failed: %s", exc)
            return None

    async def promote_prompt(self, role: str, prompt_id: str) -> None:
        """Promote a candidate prompt to active, deactivating the current one."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                # Deactivate all for this role
                await session.run(
                    """
                    MATCH (p:PromptVersion {role: $role})
                    SET p.is_active = false
                    """,
                    role=role,
                )
                # Activate the new one
                await session.run(
                    """
                    MATCH (p:PromptVersion {id: $id})
                    SET p.is_active = true
                    """,
                    id=prompt_id,
                )
        except Exception as exc:
            logger.warning("Promote prompt failed: %s", exc)

    async def update_prompt_stats(
        self, prompt_id: str, score: float
    ) -> None:
        """Update usage count and rolling average score for a prompt."""
        if not self._available:
            return
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MATCH (p:PromptVersion {id: $id})
                    SET p.score_avg = CASE
                            WHEN p.usage_count = 0 THEN $score
                            ELSE (p.score_avg * p.usage_count + $score) / (p.usage_count + 1)
                        END,
                        p.usage_count = p.usage_count + 1
                    """,
                    id=prompt_id,
                    score=score,
                )
        except Exception as exc:
            logger.warning("Update prompt stats failed: %s", exc)
