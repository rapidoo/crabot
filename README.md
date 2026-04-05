<div align="center">

# 🦀 Crabot

### Un agent IA autonome, 100% local, forgé en Bretagne.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma_4-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests: 206+](https://img.shields.io/badge/Tests-206+-green.svg)]()

**Zéro cloud. Zéro API payante. Zéro tracking.**<br>
Juste un LLM local, un crabe têtu, et ton terminal.

<img src="https://img.shields.io/badge/Made_in-Bretagne_🇫🇷-black?style=for-the-badge" alt="Made in Bretagne"/>

</div>

---

## 🎬 Démo en 30 secondes

```bash
make chat
```

```
🦀 Crabot — Agent IA local
Tape ta question ou /help pour les commandes. /quit pour sortir.

🦀 > hi

  Salut. Qu'est-ce qu'on traite ?

🦀 > je ne sais pas encore, que me proposes-tu ?

  Quelques pistes :
  - Je peux chercher les dernières news tech sur le web
  - Écrire ou analyser du code Python
  - Lire et résumer des fichiers locaux
  - Me créer de nouveaux outils à la volée
  Dis-moi ce qui t'intéresse, je m'adapte.

🦀 > Recherche les dernières news sur l'IA locale

  ✓ Search results for 'IA locale 2026'
  Voici ce que j'ai trouvé :
  1. Gemma 4 — Google sort ses modèles MoE en open weight...
  2. Ollama 0.8 — Support natif des function calls...
  3. Local-first AI — La tendance qui s'accélère...

🦀 > Crée un outil qui convertit des EUR en USD au taux 1.08

  ✓ Tool 'convert_eur_to_usd' created and registered.
    File: agent/tools/custom/convert_eur_to_usd.py
    Available immediately — persists across restarts.

🦀 > /tools
  code            Execute Python code in an isolated subprocess.
  file            Read or write files on the local filesystem.
  search          Search local files for matching content.
  web_search      Search the internet for current information.
  tool_create     Create a new persistent tool.
  convert_eur_to_usd  Convert EUR to USD at rate 1.08.  ← tout juste créé

🦀 > /quit
Bye.
```

> Pas de blabla, pas de "Bien sûr ! 😊". Crabot est direct — c'est un crabe breton, pas un chatbot corporate.

---

## 💡 Pourquoi Crabot ?

| Problème | Solution Crabot |
|----------|----------------|
| Les agents IA dépendent d'APIs cloud payantes | **100% local** — Ollama + Gemma 4, rien ne sort de ta machine |
| Les chatbots oublient tout entre les sessions | **Mémoire persistante** — Neo4j graph avec épisodes, entités, skills |
| Les LLM hallucinent sans contrôle | **Critique intégrée** — un modèle dédié note chaque réponse et force un retry si besoin |
| Les agents sont des boîtes noires | **Personnalité ouverte** — édite 5 fichiers Markdown pour tout changer |
| Les outils sont figés | **Auto-évolution** — Crabot se crée ses propres outils à la volée |

---

## 🔥 Propulsé par Gemma 4

Crabot tire parti de la **famille Gemma 4** de Google — les modèles open-weight les plus efficaces pour tourner en local :

```yaml
models:
  triage:    gemma4           # 4B — triage ultra-rapide (<1s)
  planner:   gemma4:26b      # 26B MoE — planification intelligente
  executor:  gemma4:26b      # 26B MoE — exécution des tâches
  critic:    gemma4:31b      # 31B Dense — critique exigeante
```

**Pourquoi Gemma 4 ?**
- **MoE 26B** : performance de gros modèle, consommation de petit — tourne sur un Mac M1 16 Go
- **Dense 31B** : quand il faut un regard critique sans compromis
- **4B** : triage en moins d'une seconde, zéro latence perçue
- **Open weights** : pas de licence restrictive, pas de call home

> 🧪 **Testé sur MacBook Pro M4 Pro, 48 Go RAM** — les 3 modèles Gemma 4 tournent simultanément sans broncher.

> Fonctionne aussi avec n'importe quel modèle Ollama — Llama 3, Mistral, Qwen... Change 4 lignes dans `config.yaml`.

---

## 🏗️ Architecture

```
User Input (CLI / Telegram / REPL)
    │
    ▼
Triage (4B, <1s) ──── simple ──► Réponse directe + SOUL.md
    │ complex
    ▼
Memory Read (Neo4j) ── épisodes passés + skills apprises
    │
    ▼
Planner (26B) ──────── plan JSON + AGENTS.md rules + tools dynamiques
    │
    ▼
Executor (26B) ──────── exécute les steps :
    │   • code       → subprocess Python sandboxé
    │   • file       → lecture/écriture fichiers
    │   • search     → recherche locale
    │   • web_search → Brave Search API
    │   • tool_create → crée un nouveau tool persistant
    │   • parallel via asyncio.gather
    │   • loop detection (3 détecteurs)
    ▼
Critic (31B) ────────── score 0-10, retry si < seuil adaptatif
    │
    ▼
Memory Write (Neo4j) ── épisode + entités + skills extraites
```

Le pattern **Plan → Execute → Critique** garantit que chaque réponse complexe est vérifiée avant d'être envoyée. Si le Critic n'est pas satisfait, l'Executor recommence — automatiquement.

---

## 🚀 Quickstart

### Prérequis

- **Python 3.11+**
- **Ollama** installé et lancé → [ollama.com](https://ollama.com)

```bash
# Télécharge les modèles Gemma 4
ollama pull gemma4          # 4B — minimum pour démarrer (9.6 Go)
ollama pull gemma4:26b      # 26B MoE — recommandé (18 Go)
ollama pull gemma4:31b      # 31B Dense — optionnel, pour le Critic (20 Go)
```

### Installation

```bash
git clone https://github.com/flebris/crabot.git && cd crabot
make install

# Configure tes secrets (optionnel)
cp .env.example .env
# Édite .env : TELEGRAM_BOT_TOKEN, NEO4J_PASSWORD, BRAVE_API_KEY
```

### 3 modes d'utilisation

```bash
# 💬 Mode interactif — le plus fun
make chat

# ⚡ Mode one-shot — une question, une réponse
make run PROMPT="Calcule pi avec 10 décimales"

# 🤖 Mode daemon — bot Telegram 24/7
make daemon
```

---

## 🧠 Ce qui rend Crabot intelligent

### Mémoire persistante

Crabot se souvient. Grâce à Neo4j, chaque conversation est stockée sous forme d'épisodes dans un graphe de connaissances. Les entités sont extraites, les skills sont apprises, les patterns réutilisés.

### Auto-évolution

Demande-lui de créer un outil — il le code, le teste, l'enregistre, et l'utilise immédiatement :

```
🦀 > Crée un outil qui vérifie si un site web est en ligne

  ✓ Tool 'check_website_status' created and registered.
    File: agent/tools/custom/check_website_status.py
    Available immediately and persists across restarts.
```

Les outils custom sont auto-découverts au démarrage et injectés dans le Planner.

### Dreaming (consolidation mémorielle)

Inspiré des neurosciences : Crabot consolide sa mémoire périodiquement en fonction de la fréquence, pertinence, diversité et récence des épisodes. Les souvenirs inutiles s'estompent, les patterns importants se renforcent.

### Loop detection

3 détecteurs empêchent l'agent de tourner en boucle :
- **Repeat** — détecte les réponses identiques
- **Circuit breaker** — coupe après N échecs consécutifs
- **Ping-pong** — détecte les allers-retours stériles entre Planner et Executor

### Adaptive threshold

Le seuil de qualité du Critic se calibre automatiquement sur la distribution des scores passés. Plus Crabot s'améliore, plus il est exigeant avec lui-même.

---

## 🦀 Personnalité configurable

Crabot a une âme — et tu peux la modifier. Tout est dans `workspace/` :

| Fichier | Ce qu'il contrôle | Injecté dans |
|---------|-------------------|-------------|
| `SOUL.md` | Ton, opinions, humour, limites | Executor |
| `IDENTITY.md` | Nom, emoji, creature, vibe | Telegram |
| `AGENTS.md` | Règles opérationnelles | Planner |
| `USER.md` | Profil utilisateur | Executor |
| `HEARTBEAT.md` | Tâches proactives (cron) | Daemon |

Envie d'un agent sarcastique ? Poétique ? Ultra-formel ? Change `SOUL.md` et relance.

---

## 🤖 Mode Telegram (24/7)

Crabot peut tourner en daemon et répondre sur Telegram en continu :

```bash
# En service macOS (auto-start au boot, auto-restart on crash)
make service-install && make service-start
```

- Répond à tes messages via le pipeline complet Plan → Execute → Critique
- **Heartbeat** toutes les 30 min — tâches proactives définies dans `HEARTBEAT.md`
- **Goal engine** toutes les 15 min — suit tes objectifs long terme
- Commandes Telegram : `/good`, `/bad`, `/stats`, `/help`

---

## ⚙️ Configuration

Tout dans `agent/config.yaml`. Secrets dans `.env`.

```env
# .env — rien n'est obligatoire sauf Ollama
TELEGRAM_BOT_TOKEN=ton_token_botfather     # optionnel — pour le mode Telegram
NEO4J_PASSWORD=ton_password_neo4j          # optionnel — pour la mémoire persistante
BRAVE_API_KEY=ta_cle_brave_search          # optionnel — pour la recherche web
```

> **Le minimum pour démarrer : Ollama + `gemma4`. C'est tout.** Le reste est optionnel et s'active progressivement.

---

## 🧪 Tests

206+ tests. On ne rigole pas avec la qualité.

```bash
make test                 # Unit + intégration (sans Neo4j)
make test-unit            # Unit seuls — pas besoin d'Ollama
make test-integration     # Avec Ollama réel
make test-neo4j           # Mémoire graph
make test-all             # Tout d'un coup
make benchmark            # 20 prompts avec métriques
```

---

## 📋 Toutes les commandes

```bash
make install              # Crée venv + installe tout
make chat                 # REPL interactif
make run PROMPT="..."     # One-shot
make daemon               # Bot Telegram foreground
make service-install      # Service macOS (LaunchAgent)
make service-start        # Démarre le service
make service-stop         # Arrête
make service-restart      # Redémarre
make service-status       # Statut + logs
make service-logs         # Tail -f des logs
make service-uninstall    # Supprime le service
make lint                 # mypy strict
make setup-neo4j          # Neo4j via Docker
make stop-neo4j           # Arrête Neo4j
make clean                # Supprime venv, caches, logs
```

---

## 📁 Structure du projet

```
crabot/
├── workspace/                 🦀 Personnalité
│   ├── SOUL.md                    Ton, humour, opinions
│   ├── IDENTITY.md                Nom, emoji, vibe
│   ├── AGENTS.md                  Règles opérationnelles
│   ├── USER.md                    Profil utilisateur
│   └── HEARTBEAT.md               Tâches proactives
├── agent/
│   ├── __main__.py            CLI : REPL, one-shot, daemon
│   ├── agent.py               Orchestrateur principal
│   ├── config.yaml            Configuration modèles
│   ├── schemas.py             Modèles Pydantic
│   ├── daemon.py              Telegram + scheduler
│   ├── personality/
│   │   └── loader.py          Charge workspace/
│   ├── core/
│   │   ├── triage.py          Classification simple/complex
│   │   ├── planner.py         Plan JSON structuré
│   │   ├── executor.py        Exécution (parallel, loop detection)
│   │   ├── critic.py          Scoring + retry adaptatif
│   │   ├── heartbeat.py       Tâches proactives
│   │   ├── goal_engine.py     Goals persistants
│   │   ├── scheduler.py       Cron + self-scheduling
│   │   └── reflection.py      Réflexion sur métriques
│   ├── models/
│   │   ├── ollama_client.py   Client async Ollama
│   │   └── router.py          Sélection modèle/rôle
│   ├── tools/
│   │   ├── registry.py        Auto-discovery
│   │   ├── code_exec.py       Sandbox Python
│   │   ├── file_io.py         Fichiers (chemin restreint)
│   │   ├── search.py          Recherche locale
│   │   ├── web_search.py      Brave Search API
│   │   ├── tool_create.py     Meta-tool : auto-création
│   │   └── custom/            🔧 Outils générés par l'agent
│   ├── intelligence/
│   │   ├── loop_detection.py  3 détecteurs de boucles
│   │   ├── dreaming.py        Consolidation mémorielle
│   │   └── skill_injector.py  Injection skills
│   ├── memory/
│   │   ├── neo4j_client.py    Graphe (épisodes, entités, skills)
│   │   ├── entity_extractor.py Extraction entités via LLM
│   │   └── schemas.cypher     Setup Neo4j
│   └── infra/
│       ├── state.py           Crash recovery
│       ├── retry.py           Backoff exponentiel
│       └── metrics.py         Métriques JSONL
├── tests/                     206+ tests
└── benchmarks/                Suite benchmark (20 prompts)
```

---

## 🗺️ Roadmap

### 🔐 Sécurité & authentification
- [ ] **Auth Telegram renforcée** — whitelist par username en plus des user IDs, confirmation à la première connexion
- [ ] **Rate limiting** — limite de requêtes par utilisateur (anti-flood / anti-abus)
- [ ] **Sandboxing renforcé** — isolation des outils custom (seccomp / nsjail pour `code_exec`)
- [ ] **Audit log** — journal immuable de toutes les actions agent (qui, quand, quoi)
- [ ] **Chiffrement mémoire** — chiffrement at-rest des épisodes Neo4j (données personnelles)

### 👥 Multi-utilisateurs
- [ ] **Contextes isolés** — chaque utilisateur a sa propre mémoire, ses goals, ses outils custom
- [ ] **Profils utilisateur dynamiques** — `USER.md` par utilisateur, appris au fil des conversations
- [ ] **Rôles & permissions** — admin / utilisateur / lecture seule
- [ ] **Quotas** — limites de tokens / requêtes par utilisateur et par jour

### 🧠 Intelligence
- [ ] **RAG sur documents locaux** — ingestion PDF, Markdown, code source avec chunking + embeddings
- [ ] **Conversation multi-tours** — contexte glissant sur les N derniers échanges
- [ ] **Planning long terme** — décomposition de projets en sous-goals avec suivi automatique
- [ ] **Self-evaluation benchmarks** — l'agent s'auto-évalue sur une suite de tests et ajuste ses prompts
- [ ] **Fine-tuning local** — adaptation du modèle sur les épisodes à score élevé (LoRA)

### 🔌 Intégrations & interfaces
- [ ] **Interface web locale** — dashboard avec historique, métriques, configuration en live
- [ ] **API REST** — endpoint HTTP pour intégrer Crabot dans d'autres outils
- [ ] **Discord / Slack** — interfaces alternatives à Telegram
- [ ] **Webhooks** — notifications push sur événements (goal atteint, erreur critique, etc.)
- [ ] **MCP server** — exposer Crabot comme serveur Model Context Protocol

### 📦 Distribution
- [ ] **Docker Compose** — one-liner avec Ollama + Neo4j + Crabot
- [ ] **Homebrew tap** — `brew install crabot`
- [ ] **Plugins communautaires** — marketplace de `agent/tools/custom/` partagés
- [ ] **Config wizard** — assistant interactif de première installation
- [ ] **Documentation multilingue** — README en anglais, docs en FR/EN

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Crabot est un projet open source né en Bretagne, mais ouvert au monde.

1. Fork le repo
2. Crée une branche (`git checkout -b feature/mon-truc`)
3. Commite (`git commit -m 'Add mon truc'`)
4. Push (`git push origin feature/mon-truc`)
5. Ouvre une Pull Request

---

## 📜 Licence

Apache 2.0 — libre d'utilisation, modification et distribution. Protection brevets incluse pour les contributeurs.

Voir le fichier [LICENSE](LICENSE) pour les détails.

---

<div align="center">

**Crafté avec obstination en Bretagne 🦀🌊**<br>
<sub>Par <a href="https://github.com/flebris">Frédéric Le Bris</a> — Le Bris Consulting</sub><br>
<sub>Propulsé par <a href="https://ollama.com">Ollama</a> et <a href="https://ai.google.dev/gemma">Gemma 4</a></sub>

*Aucun crabe n'a été maltraité pendant le développement de ce projet.*

</div>
