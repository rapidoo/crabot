# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano is a **100% local AI agent** in Python, using Ollama (no cloud dependencies, no third-party SDKs). It implements a **Plan → Execute → Critique** pattern that compensates for weaker local models with a strong scaffold architecture.

Target hardware: Apple Silicon M4 Pro, 48 GB unified memory.

## Architecture

The agent uses a multi-model routing strategy across Gemma 4 variants:
- **E4B** (9.6 GB): fast triage, draft generation, classification (thinking off)
- **26B MoE** (18 GB): main backbone for planning and execution (thinking on)
- **31B Dense** (20 GB): rigorous critic/evaluation (thinking on)

RAM constraint: only 2 models fit simultaneously. Ollama handles model swapping (~5s penalty).

### Core Pipeline

1. **Triage** (E4B) — classify input complexity: simple | complex | multimodal
2. **Memory Read** — enrich context from Neo4j graph (past episodes, entities)
3. **Plan** (26B) — produce structured JSON plan with steps and tools
4. **Execute** (26B) — run each step, call tools, use Skeleton-of-Thought for long outputs
5. **Critique** (31B) — score each step 0-10, retry if < 6.5 (max 3 retries)
6. **Memory Write** — persist episode, entities, and scores to Neo4j

### Key Design Decisions

- All LLM calls go directly to Ollama `/api/chat` — no SDK wrapper
- Plans are validated with **pydantic** (strict JSON schema)
- Tool registry uses factory pattern — add tools without touching the orchestrator
- Cursor recovery: persist step progress for crash resilience (atomic write .tmp + rename)
- Skeleton-of-Thought activated when expected output > 500 tokens
- Thinking mode via `<|think|>` token — enabled for Planner/Critic, disabled for Executor

## Target Project Structure

```
agent/
  core/          # planner.py, executor.py, critic.py
  memory/        # neo4j_client.py, schemas.cypher, entity_extractor.py
  tools/         # registry.py, search.py, code_exec.py, file_io.py
  models/        # ollama_client.py, router.py
  infra/         # state.py (cursor recovery), retry.py (exponential backoff)
  agent.py       # Main orchestrator
  config.yaml    # All configurable parameters
```

## Key Technical Constraints

- **Zero cloud dependency**: all inference local via Ollama
- **Async throughout**: `asyncio` for parallel step execution and tool calls
- **Neo4j** for persistent episodic memory (graph schema: Episode, Entity, Skill nodes)
- **Subprocess sandbox** for code execution tool (timeout + isolation)
- Sampling params per role: Planner (temp 0.7), Critic (temp 0.3), default (temp 1.0)

## Reference Documents

- `AGENT_PATTERN.md` — full architecture spec, prompts, code contracts, and config
- `docs/PLAN_ENRICHISSEMENT.md` — enrichment plan (NanoClaw patterns, Gemma 4 specs)
- `gemma4.html` — Gemma 4 model documentation

## Language

Documentation is in French. Code and technical terms in English. Comments in code should be in English.
