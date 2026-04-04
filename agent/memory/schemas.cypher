// Neo4j schema setup — constraints and indexes for the agent memory graph.
// Run once at initial setup: cat schemas.cypher | cypher-shell -u neo4j -p <password>

// Constraints
CREATE CONSTRAINT episode_id IF NOT EXISTS
FOR (ep:Episode) REQUIRE ep.id IS UNIQUE;

CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT skill_name IF NOT EXISTS
FOR (s:Skill) REQUIRE s.name IS UNIQUE;

// Indexes for query performance
CREATE INDEX episode_score IF NOT EXISTS
FOR (ep:Episode) ON (ep.score);

CREATE INDEX episode_timestamp IF NOT EXISTS
FOR (ep:Episode) ON (ep.timestamp);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);
