# Nano Agent

Agent IA 100% local utilisant Ollama. Pattern **Plan → Execute → Critique** avec scaffold fort pour compenser les modèles locaux.

Zéro dépendance cloud, zéro SDK tiers — uniquement des appels HTTP directs à Ollama.

## Prérequis

1. **Python 3.11+**
2. **Ollama** installé et lancé ([ollama.com](https://ollama.com))
3. **Au moins un modèle Gemma 4** :
   ```bash
   ollama pull gemma4          # E4B, 9.6 GB — minimum requis
   ollama pull gemma4:31b      # 31B Dense, 20 GB — optionnel, pour le Critic
   ```
4. **Neo4j** (optionnel) — pour la mémoire épisodique entre sessions

## Installation

```bash
git clone <repo-url> && cd nano

# Crée le venv et installe tout
make install

# Vérifier que ça marche
make test-unit    # 144 tests, pas besoin d'Ollama
```

## Utilisation

### Lancer l'agent

```bash
# Syntaxe de base
.venv/bin/python -m agent "ta question ou tâche ici"

# Raccourci via Makefile
make run PROMPT="What is 2+2?"
```

### Exemples concrets

```bash
# Question simple → triage E4B, réponse directe en ~1s
.venv/bin/python -m agent "What is the capital of France?"

# Tâche complexe → Plan + Execute + Critique (~20s)
.venv/bin/python -m agent "Search for all Python files in the agent directory and count them"

# Génération de code → Plan + code_exec sandbox + Critique
.venv/bin/python -m agent "Write a Python function to compute Fibonacci numbers"
```

### Ce qui se passe sous le capot

```
User Input
    │
    ▼
Triage (E4B, <1s) ──simple──► Réponse directe E4B
    │ complex
    ▼
Memory Read (Neo4j) ── injecte le contexte des épisodes passés
    │
    ▼
Planner (26B) ── produit un plan JSON structuré [{step, tool, input}]
    │
    ▼
Executor (26B) ── exécute chaque step :
    │   • tool=code  → subprocess Python sandboxé (timeout 30s)
    │   • tool=file  → lecture/écriture fichiers (chemin restreint)
    │   • tool=search → recherche keyword dans les fichiers locaux
    │   • tool=none  → génération LLM directe
    │   • steps parallèles via asyncio.gather
    ▼
Critic (31B) ── note chaque step 0-10, retry automatique si < 6.5
    │
    ▼
Memory Write (Neo4j) ── persiste l'épisode + entités extraites
```

### Lire les logs

L'agent produit des logs en temps réel dans le terminal et une trace JSONL :

```bash
# Voir la trace des exécutions passées
cat logs/agent_trace.jsonl | python -m json.tool

# Chaque entrée contient :
# input, goal, steps, scores[], elapsed_s, path (simple/complex), memory (true/false)
```

## Configuration

Tout est dans **`agent/config.yaml`** — édite ce fichier pour adapter l'agent :

### Choisir les modèles

```yaml
models:
  triage:    gemma4          # Modèle rapide pour le triage (E4B)
  planner:   gemma4          # Modèle pour la planification
  executor:  gemma4          # Modèle pour l'exécution
  critic:    gemma4:31b      # Modèle lourd pour la critique (optionnel)
  critic_light: gemma4       # Fallback si le 31B n'est pas dispo
```

> **Astuce** : si tu n'as que `gemma4` (E4B), l'agent fonctionne — il utilisera le même modèle partout. Les scores Critic seront moins fiables mais le pipeline tourne.

### Ajuster les seuils

```yaml
thresholds:
  min_score: 6.5     # En dessous, le Critic déclenche un retry
  max_retries: 3     # Nombre max de retries par step

skeleton:
  enabled: true
  token_threshold: 500  # Active Skeleton-of-Thought pour les sorties longues
```

### Activer Neo4j (mémoire entre sessions)

```yaml
memory:
  neo4j_uri: bolt://localhost:7687
  neo4j_user: neo4j
  neo4j_password: ton_password
  context_k: 5           # Nombre d'épisodes passés injectés dans le Planner
  write_threshold: 5.0   # Ne persiste que les épisodes avec score > 5
```

Sans Neo4j, l'agent fonctionne normalement — il n'a juste pas de mémoire entre sessions.

## Neo4j (optionnel)

```bash
# Lancer Neo4j via Docker
make setup-neo4j

# Ou manuellement
docker run -d -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/ton_password neo4j:5

# Mettre le password dans agent/config.yaml → memory.neo4j_password

# Tester la connexion
make test-neo4j

# Visualiser le graphe : http://localhost:7474

# Arrêter
make stop-neo4j
```

L'agent persiste automatiquement chaque exécution (goal, summary, score, entités extraites). Au run suivant, il injecte les épisodes pertinents dans le prompt du Planner.

## Telegram Bot

L'agent peut tourner comme bot Telegram 24/7.

### Setup

1. **Créer un bot** : ouvre Telegram, cherche @BotFather, envoie `/newbot`, suis les instructions. Tu recevras un token type `7123456789:AAH...`.

2. **Configurer le token** dans `agent/config.yaml` :
   ```yaml
   telegram:
     bot_token: "7123456789:AAHxxxxx"   # Ton token BotFather
     allowed_users: []                   # Vide = tout le monde peut parler au bot
     # allowed_users: [123456789]        # Ou restreindre à ton user ID
   ```
   Ou via variable d'environnement : `export TELEGRAM_BOT_TOKEN="7123456789:AAHxxxxx"`

3. **Trouver ton user ID** (optionnel, pour `allowed_users`) : envoie un message à @userinfobot sur Telegram.

### Lancer en foreground (test)

```bash
# Mode daemon en avant-plan (Ctrl+C pour arrêter)
make daemon

# Ou directement
python -m agent --daemon
```

Envoie un message au bot depuis Telegram — il répond avec le résultat du pipeline.

### Lancer comme service macOS (24/7)

```bash
# Installer le service launchd
make service-install

# Démarrer (auto-restart si crash, se relance au login)
make service-start

# Vérifier le statut
make service-status

# Voir les logs en live
make service-logs

# Arrêter
make service-stop

# Redémarrer
make service-restart

# Désinstaller complètement
make service-uninstall
```

Le service :
- Se lance automatiquement au login macOS
- Redémarre automatiquement si le process crash
- Logs dans `logs/daemon-stdout.log` et `logs/daemon-stderr.log`
- Throttle de 10s entre les restart pour éviter les boucles

### Fonctionnement du bot

- **Questions simples** : réponse rapide (~1s) via le triage E4B
- **Tâches complexes** : pipeline complet Plan → Execute → Critique
- **Indicateur "typing..."** pendant le traitement
- **Messages longs** découpés automatiquement (limite Telegram 4096 chars)
- **Concurrence** : max 3 requêtes simultanées (configurable dans `daemon.max_concurrent`)
- **Timeout** : 5 min par requête (configurable dans `daemon.request_timeout`)

## Commandes make

```bash
# Installation
make install          # Crée venv + installe deps (y compris telegram)

# Tests
make test             # Tests unit + intégration Ollama (sans Neo4j)
make test-unit        # Tests unitaires seuls (pas besoin d'Ollama)
make test-integration # Tests avec Ollama réel
make test-neo4j       # Tests mémoire graph
make test-all         # Tout (unit + intégration + neo4j)
make lint             # mypy
make benchmark        # 20 prompts variés avec métriques

# Agent
make run PROMPT="..."   # One-shot
make daemon             # Bot Telegram en foreground

# Service macOS
make service-install    # Installe le LaunchAgent
make service-start      # Démarre le daemon
make service-stop       # Arrête le daemon
make service-restart    # Redémarre
make service-status     # Statut + derniers logs
make service-logs       # Tail -f des logs
make service-uninstall  # Supprime le service

# Infra
make setup-neo4j      # Lance Neo4j via Docker
make stop-neo4j       # Arrête Neo4j
make clean            # Supprime venv, caches, logs
```

## Crash recovery

Si l'agent plante en pleine exécution (kill, OOM, Ctrl+C) :

- L'état est sauvegardé step par step dans `state/agent_state.json`
- Au prochain lancement, l'agent détecte l'état et **reprend au dernier step réussi**
- Écriture atomique (.tmp + rename) — pas de corruption possible

Pour forcer un reset :
```bash
rm state/agent_state.json
```

## Ajouter un outil

Créer un fichier dans `agent/tools/`, implémenter le protocole Tool, et appeler `register_tool` :

```python
# agent/tools/mon_outil.py
from agent.tools.registry import register_tool

class MonOutil:
    @property
    def name(self) -> str: return "mon_outil"

    @property
    def description(self) -> str: return "Description de mon outil"

    async def run(self, input: str) -> str:
        # ... logique ici
        return "résultat"

register_tool("mon_outil", lambda: MonOutil())
```

Puis ajouter l'import dans `agent/core/executor.py` :
```python
import agent.tools.mon_outil  # noqa: F401
```

Le Planner pourra utiliser `"tool": "mon_outil"` dans ses plans (ajouter le nom dans le system prompt du Planner dans `agent/core/planner.py`).

## Structure du projet

```
nano/
  agent/
    config.yaml         Configuration (modèles, seuils, Neo4j)
    config.py           Loader pydantic
    schemas.py          Modèles de données (Plan, Step, CriticScore...)
    agent.py            Orchestrateur principal
    __main__.py         CLI : python -m agent "prompt"
    core/
      triage.py         Classification rapide (simple/complex/multimodal)
      planner.py        Génère un plan JSON structuré
      executor.py       Exécute les steps (tools + LLM + parallel + SoT)
      critic.py         Note et retry
    models/
      ollama_client.py  Client HTTP async pour Ollama /api/chat
      router.py         Sélection du modèle par rôle
    tools/
      registry.py       Factory pattern pour les outils
      code_exec.py      Exécution Python sandboxée (subprocess + timeout)
      file_io.py        Lecture/écriture fichiers (chemin restreint)
      search.py         Recherche keyword dans les fichiers
    memory/
      neo4j_client.py   Mémoire graph épisodique
      entity_extractor.py  Extraction d'entités via LLM
      schemas.cypher    Setup Neo4j (contraintes + index)
    infra/
      state.py          Cursor recovery (crash-safe)
      retry.py          Backoff exponentiel
  tests/                162 tests (unit + intégration + Neo4j)
  benchmarks/           Suite de benchmark (20 prompts)
  AGENT_PATTERN.md      Document d'architecture détaillé
```

## Multi-modèle Gemma 4

| Variante | RAM (Q4) | Rôle dans l'agent |
|----------|----------|-------------------|
| **E4B** | 9.6 GB | Triage (<1s), draft, entity extraction |
| **26B MoE** | 18 GB | Planner + Executor (backbone) |
| **31B Dense** | 20 GB | Critic (évaluation rigoureuse) |

Sur un M4 Pro 48 GB : le 26B (18 GB) + E4B (9.6 GB) tiennent ensemble. Le 31B nécessite de décharger le 26B (~5s de swap géré par Ollama).

Avec seulement `gemma4` (E4B) : l'agent tourne, mais les plans sont moins fiables. Le scaffold (retry, critic) compense en partie.
