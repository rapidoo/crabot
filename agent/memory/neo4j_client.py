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
