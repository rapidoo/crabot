# 🦀 Crabot — Agent IA Autonome Local

Agent IA 100% local, autonome et auto-évolutif. Utilise Ollama (Gemma 4) avec un pattern **Plan → Execute → Critique** et une personnalité configurable.

Zéro dépendance cloud — appels HTTP directs à Ollama, mémoire Neo4j, recherche Brave.

## Prérequis

1. **Python 3.11+**
2. **Ollama** installé et lancé ([ollama.com](https://ollama.com))
3. **Modèles Gemma 4** :
   ```bash
   ollama pull gemma4          # E4B, 9.6 GB — minimum
   ollama pull gemma4:26b      # 26B MoE, 18 GB — recommandé (planner/executor)
   ollama pull gemma4:31b      # 31B Dense, 20 GB — optionnel (critic)
   ```
4. **Neo4j** (optionnel) — mémoire épisodique entre sessions
5. **Brave Search API** (optionnel) — recherche internet

## Installation

```bash
git clone <repo-url> && cd nano
make install

# Configurer les secrets
cp .env.example .env
# Éditer .env : TELEGRAM_BOT_TOKEN, NEO4J_PASSWORD, BRAVE_API_KEY
```

## 3 modes d'utilisation

### Mode interactif (REPL)

```bash
python -m agent
# ou
make chat
```

```
🦀 Crabot — Agent IA local
Tape ta question ou /help pour les commandes. /quit pour sortir.

🦀 > Recherche les dernières news sur l'IA locale
  ✓ Search results for 'IA locale 2026'...

🦀 > /tools
  code            Execute Python code in an isolated subprocess.
  file            Read or write files on the local filesystem.
  search          Search local files for matching content.
  web_search      Search the internet for current information.
  tool_create     Create a new persistent tool.

🦀 > /goals
  [3] Apprendre Rust

🦀 > /good
  Noted — positive feedback saved.

🦀 > /quit
```

Commandes REPL : `/tools`, `/stats`, `/goals`, `/goal <desc>`, `/good`, `/bad [raison]`, `/quit`

### Mode one-shot

```bash
python -m agent "What is the capital of France?"
make run PROMPT="Calcule pi avec 10 décimales"
```

### Mode daemon — Telegram 24/7

```bash
# Foreground (test)
make daemon

# Service macOS (auto-start, auto-restart)
make service-install && make service-start
```

Le bot Telegram :
- Répond à tes messages via le pipeline complet
- Commandes : `/good`, `/bad`, `/stats`, `/help`
- Heartbeat toutes les 30 min (tâches proactives)
- Goal engine toutes les 15 min

## Architecture

```
User Input (CLI / Telegram / REPL)
    │
    ▼
Triage (E4B, <1s) ──simple──► Réponse directe + SOUL.md
    │ complex
    ▼
Memory Read (Neo4j) ── épisodes passés + skills apprises
    │
    ▼
Planner (26B) ── plan JSON + AGENTS.md rules + tools dynamiques
    │
    ▼
Executor (26B) ── exécute les steps :
    │   • code       → subprocess Python sandboxé
    │   • file       → lecture/écriture fichiers
    │   • search     → recherche locale keyword
    │   • web_search → Brave Search API
    │   • tool_create → crée un nouveau tool persistant
    │   • none       → génération LLM directe
    │   • parallel via asyncio.gather
    │   • loop detection (repeat/circuit breaker/ping-pong)
    ▼
Critic (31B) ── score 0-10, retry si < seuil adaptatif
    │
    ▼
Memory Write (Neo4j) ── épisode + entités + skills extraites
```

## Personnalité (workspace/)

L'agent a une identité configurable via des fichiers Markdown :

| Fichier | Rôle | Injecté dans |
|---------|------|-------------|
| `SOUL.md` | Ton, opinions, humour, limites | Executor (réponses directes) |
| `IDENTITY.md` | Nom, emoji, creature, vibe | Telegram (préfixe messages) |
| `AGENTS.md` | Règles opérationnelles | Planner (system prompt) |
| `USER.md` | Profil utilisateur | Executor (adaptation) |
| `HEARTBEAT.md` | Tâches proactives (toutes les 30 min) | Daemon (scheduler) |

Pour personnaliser : édite les fichiers dans `workspace/`.

## Auto-évolution

L'agent peut **se créer de nouveaux outils** qui persistent :

```
🦀 > Crée un outil qui convertit des EUR en USD au taux 1.08
  ✓ Tool 'convert_eur_to_usd' created and registered.
    File: agent/tools/custom/convert_eur_to_usd.py
    Available immediately and persists across restarts.
```

- Les tools sont auto-découverts au démarrage (`agent/tools/` + `agent/tools/custom/`)
- Le Planner construit sa liste d'outils dynamiquement depuis le registry
- Les skills (chaînes d'outils réussies) sont extraites et réinjectées dans les futurs plans

## Intelligence

| Feature | Description |
|---------|------------|
| **Adaptive threshold** | Le seuil du Critic se calibre sur la distribution des scores passés |
| **Loop detection** | 3 détecteurs : repeat, circuit breaker, ping-pong |
| **Skill injection** | Réutilise les patterns d'outils éprouvés dans le Planner |
| **Dreaming** | Consolidation mémorielle bio-inspirée (fréquence/pertinence/diversité/récence) |
| **Reflection** | Analyse périodique des métriques → ajustements automatiques |
| **Metrics** | Collecte scores, latence, taux retry, usage outils depuis la trace JSONL |

## Configuration

Tout dans `agent/config.yaml`. Secrets dans `.env`.

```yaml
models:
  triage:    gemma4           # E4B pour le triage rapide
  planner:   gemma4:26b      # 26B MoE pour la planification
  executor:  gemma4:26b      # 26B MoE pour l'exécution
  critic:    gemma4:31b      # 31B Dense pour la critique
  critic_light: gemma4:26b   # Fallback si 31B absent
```

```env
# .env
TELEGRAM_BOT_TOKEN=ton_token_botfather
NEO4J_PASSWORD=ton_password_neo4j
BRAVE_API_KEY=ta_cle_brave_search
```

## Commandes make

```bash
# Installation
make install              # Crée venv + installe tout

# Agent
make chat                 # REPL interactif
make run PROMPT="..."     # One-shot
make daemon               # Bot Telegram foreground

# Service macOS (24/7)
make service-install      # Installe le LaunchAgent
make service-start        # Démarre
make service-stop         # Arrête
make service-restart      # Redémarre
make service-status       # Statut + logs
make service-logs         # Tail -f des logs
make service-uninstall    # Supprime

# Tests (206+)
make test                 # Unit + intégration (sans Neo4j)
make test-unit            # Unit seuls (pas besoin d'Ollama)
make test-integration     # Avec Ollama réel
make test-neo4j           # Mémoire graph
make test-all             # Tout

# Outils
make lint                 # mypy
make benchmark            # 20 prompts avec métriques
make setup-neo4j          # Neo4j via Docker
make stop-neo4j           # Arrête Neo4j
make clean                # Supprime venv, caches, logs
```

## Structure

```
nano/
  workspace/              Personnalité (SOUL.md, IDENTITY.md, AGENTS.md, USER.md, HEARTBEAT.md)
  agent/
    __main__.py           CLI : REPL, one-shot, daemon
    agent.py              Orchestrateur principal
    config.yaml           Configuration
    schemas.py            Modèles pydantic (Plan, Step, CriticScore...)
    env.py                Loader .env
    daemon.py             Daemon Telegram + scheduler
    personality/
      loader.py           Charge les fichiers workspace/
    core/
      triage.py           Classification rapide (simple/complex)
      planner.py          Plan JSON structuré (tools dynamiques)
      executor.py         Exécution steps (parallel, SoT, loop detection)
      critic.py           Scoring + retry + seuil adaptatif
      heartbeat.py        Tâches proactives (HEARTBEAT.md)
      goal_engine.py      Gestion des goals persistants
      scheduler.py        Cron tasks + self-scheduling
      reflection.py       Cycle de réflexion sur les métriques
    models/
      ollama_client.py    Client async Ollama /api/chat
      router.py           Sélection modèle par rôle
    tools/
      registry.py         Auto-discovery + factory pattern
      code_exec.py        Sandbox Python (subprocess + timeout)
      file_io.py          Fichiers (chemin restreint)
      search.py           Recherche locale
      web_search.py       Brave Search API
      tool_create.py      Meta-tool : l'agent crée ses propres outils
      custom/             Outils générés par l'agent (persistants)
    intelligence/
      loop_detection.py   3 détecteurs de boucles infinies
      dreaming.py         Consolidation mémorielle bio-inspirée
      skill_injector.py   Injection skills dans le Planner
    memory/
      neo4j_client.py     Mémoire graph (épisodes, entités, skills, goals)
      entity_extractor.py Extraction entités via LLM
      schemas.cypher      Setup Neo4j
    infra/
      state.py            Crash recovery (atomic write)
      retry.py            Backoff exponentiel
      metrics.py          Collecte métriques depuis trace JSONL
  tests/                  206+ tests
  benchmarks/             Suite benchmark (20 prompts)
```
