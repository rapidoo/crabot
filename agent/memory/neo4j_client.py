"""Neo4j memory client — episodic graph memory with graceful degradation."""

from __future__ import annotations

import logging
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
        self._password = password or settings.memory.neo4j_password
        self._context_k = settings.memory.context_k
        self._write_threshold = settings.memory.write_threshold
        self._driver: Any = None
        self._available = False

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

        queries = [
            "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE INDEX episode_score IF NOT EXISTS FOR (ep:Episode) ON (ep.score)",
            "CREATE INDEX episode_timestamp IF NOT EXISTS FOR (ep:Episode) ON (ep.timestamp)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        try:
            async with self._driver.session() as session:
                for q in queries:
                    await session.run(q)
            logger.info("Neo4j schema setup complete")
        except Exception as exc:
            logger.warning("Neo4j schema setup failed: %s", exc)

    async def get_context(
        self, detected_entities: list[str]
    ) -> str:
        """Query top-K relevant episodes for the given entities.

        Returns a formatted context string for injection into the planner prompt.
        Returns empty string if Neo4j is unavailable.
        """
        if not self._available or not detected_entities:
            return ""

        query = """
        MATCH (e:Entity)<-[:HAS_ENTITY]-(ep:Episode)
        WHERE e.name IN $entities AND ep.score > 6.5
        RETURN ep.summary AS summary, ep.timestamp AS ts, ep.goal AS goal
        ORDER BY ep.timestamp DESC
        LIMIT $limit
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    query, entities=detected_entities, limit=self._context_k
                )
                records = [r async for r in result]

            if not records:
                return ""

            lines = []
            for r in records:
                ts = r["ts"] or "unknown"
                lines.append(f"- [{ts}] {r['goal']}: {r['summary']}")

            context = "Relevant past episodes:\n" + "\n".join(lines)
            logger.info("Memory: enriched context with %d episodes", len(records))
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
    ) -> str | None:
        """Persist an episode with its entities to the graph.

        Args:
            goal: The episode's goal.
            summary: Summary of what happened.
            score: Average critic score.
            entities: List of {"name": ..., "type": ..., "description": ...}.
            previous_episode_id: ID of the previous episode to chain with.

        Returns:
            The episode ID, or None if persistence failed.
        """
        if not self._available:
            return None

        if score < self._write_threshold:
            logger.debug("Score %.1f below threshold %.1f, skipping persist", score, self._write_threshold)
            return None

        episode_id = str(uuid.uuid4())[:8]

        try:
            async with self._driver.session() as session:
                # Create episode
                await session.run(
                    """
                    MERGE (ep:Episode {id: $id})
                    SET ep.timestamp = datetime(),
                        ep.summary = $summary,
                        ep.score = $score,
                        ep.goal = $goal
                    """,
                    id=episode_id,
                    summary=summary,
                    score=score,
                    goal=goal,
                )

                # Link entities
                for entity in entities:
                    await session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $description
                        WITH e
                        MATCH (ep:Episode {id: $episode_id})
                        MERGE (ep)-[:HAS_ENTITY]->(e)
                        """,
                        name=entity.get("name", ""),
                        type=entity.get("type", ""),
                        description=entity.get("description", ""),
                        episode_id=episode_id,
                    )

                # Chain with previous episode
                if previous_episode_id:
                    await session.run(
                        """
                        MATCH (ep:Episode {id: $id})
                        MATCH (prev:Episode {id: $prev_id})
                        MERGE (ep)-[:FOLLOWS]->(prev)
                        """,
                        id=episode_id,
                        prev_id=previous_episode_id,
                    )

            logger.info(
                "Memory: persisted episode %s (score=%.1f, %d entities)",
                episode_id, score, len(entities),
            )
            return episode_id

        except Exception as exc:
            logger.warning("Memory write failed: %s", exc)
            return None
