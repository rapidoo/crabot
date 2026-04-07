<div align="center">

# 🦀 Crabot

### Un agent IA autonome, 100% local, qui apprend et se modifie lui-même.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma_4_|_Mistral-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

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

🦀 > Recherche les dernières news sur l'IA locale

  Voici ce que j'ai trouvé :
  1. Gemma 4 — Google sort ses modèles MoE en open weight...
  2. Ollama 0.8 — Support natif des function calls...
  3. Local-first AI — La tendance qui s'accélère...

🦀 > Crée un outil qui convertit des EUR en USD au taux 1.08

  Tool 'convert_eur_to_usd' created and registered.
  File: agent/tools/custom/convert_eur_to_usd.py
  Available immediately — persists across restarts.

🦀 > non, utilise plutôt le taux 1.12

  Lesson learned: Utiliser le taux EUR/USD de 1.12
  Tool updated.
```

> Pas de blabla, pas de "Bien sûr ! 😊". Crabot est direct — c'est un crabe breton, pas un chatbot corporate.

---

## 💡 Pourquoi Crabot ?

| Problème | Solution Crabot |
|----------|----------------|
| Les agents IA dépendent d'APIs cloud payantes | **100% local** — Ollama + Gemma 4 / Mistral, rien ne sort de ta machine |
| Les chatbots oublient tout entre les sessions | **Mémoire persistante** — Neo4j graph avec épisodes, entités, skills, lessons |
| Les LLM hallucinent sans contrôle | **Critique intégrée** — un modèle dédié note chaque réponse et force un retry si < seuil |
| Les agents sont des boîtes noires | **Personnalité ouverte** — 5 fichiers Markdown définissent tout son comportement |
| Les outils sont figés | **Auto-évolution** — Crabot crée ses propres outils et modifie son propre code |
| Les agents ne retiennent pas les corrections | **Apprentissage continu** — chaque correction est extraite, stockée, et réinjectée |

---

## 🔥 Multi-modèles : Gemma 4 + Mistral

Crabot supporte deux familles de modèles, sélectionnables au lancement :

```bash
make chat                    # Gemma 4 (défaut)
make chat MODEL=MISTRAL      # Mistral
```

```yaml
# Gemma 4 (défaut)                    # Mistral
triage:    gemma4           # 4B       triage:    ministral-3:8b
planner:   gemma4:26b       # 26B MoE  planner:   mistral-small3.2
executor:  gemma4:26b       # 26B MoE  executor:  mistral-small3.2
critic:    gemma4:31b       # 31B      critic:    mistral-small3.2
```

> Fonctionne aussi avec n'importe quel modèle Ollama — Llama 3, Qwen... Change 4 lignes dans `config.yaml`.

> 🧪 **Testé sur MacBook Pro M4 Pro, 48 Go RAM.**

---

## 🏗️ Architecture

```
User Input (CLI / Telegram)
    │
    ▼
┌─ Supervisor ─────────────────────────────────────┐
│  Gère les interfaces, spawne un worker           │
│  redémarrable pour le rechargement de code       │
└──────────┬───────────────────────────────────────┘
           │ IPC (Unix socket)
           ▼
┌─ Execution Worker ───────────────────────────────┐
│                                                   │
│  Triage (4B) ──── simple ──► Réponse directe     │
│      │ complex                                    │
│      ▼                                            │
│  Memory Read ── recherche hybride + lessons       │
│      ▼                                            │
│  Planner (26B) ── plan JSON + rules + skills      │
│      ▼                                            │
│  Executor (26B) ── tools, code, web, parallel     │
│      ▼                                            │
│  Critic (31B) ── score 0-10, retry si < seuil     │
│      ▼                                            │
│  Memory Write ── épisode + entités + lessons       │
│      ▼                                            │
│  Evolution ── workspace, prompts, skills          │
│                                                   │
└───────────────────────────────────────────────────┘
```

Le pattern **Plan → Execute → Critique** garantit que chaque réponse complexe est vérifiée avant d'être envoyée.

> Pour la documentation architecture complète, voir [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 🧠 Ce qui rend Crabot intelligent

### Apprentissage continu

Crabot apprend de tes corrections. Quand tu le corriges ("non, utilise uv au lieu de pip"), il :
1. **Détecte** la correction automatiquement
2. **Extrait** une lesson structurée (règle, contexte, catégorie)
3. **Stocke** dans Neo4j avec un embedding vectoriel
4. **Réinjecte** les lessons pertinentes dans le Planner à chaque requête

Les lessons se renforcent quand la même correction revient. Plus une lesson est confirmée, plus elle a de poids.

### Auto-modification du code

En mode daemon, Crabot tourne avec une **architecture supervisor + worker**. Quand il modifie son propre code :
1. Un **evolution worker** applique les mutations et valide (syntax check, import check, smoke test)
2. Si tout passe, le supervisor **redémarre le worker** avec le nouveau code
3. Si ça échoue, **rollback automatique** — le service n'est jamais interrompu

### Personnalité évolutive

Les fichiers `workspace/` ne sont plus statiques. Toutes les 20 interactions, Crabot **synthétise ses lessons** et fait évoluer ses propres fichiers :
- **USER.md** — profil utilisateur enrichi par les préférences détectées
- **AGENTS.md** — règles opérationnelles ajustées par les corrections
- **SOUL.md** — personnalité affinée par le feedback

### Mémoire persistante + recherche hybride

Chaque conversation est stockée comme épisode dans Neo4j. La recherche de contexte est **hybride** : similarité vectorielle (70%) via `nomic-embed-text` + matching lexical sur entités (30%).

### Outils auto-créés

```
🦀 > Crée un outil qui vérifie si un site web est en ligne

  Tool 'check_website_status' created and registered.
  Available immediately and persists across restarts.
```

Les outils sont validés par AST, sandboxés, et auto-découverts au démarrage.

### Boucles de qualité

- **Critic adaptatif** — le seuil de qualité se calibre sur les scores passés
- **Loop detection** — 3 détecteurs (repeat, circuit breaker, ping-pong)
- **A/B testing de prompts** — les prompts évoluent et sont promus si meilleurs
- **Compression mémorielle** — les vieux épisodes à faible score sont purgés

---

## 🦀 Personnalité configurable

Tout est dans `workspace/` :

| Fichier | Ce qu'il contrôle | Évolue automatiquement ? |
|---------|-------------------|:------------------------:|
| `SOUL.md` | Ton, opinions, humour, limites | Oui |
| `AGENTS.md` | Règles opérationnelles | Oui |
| `USER.md` | Profil utilisateur | Oui |
| `IDENTITY.md` | Nom, emoji, creature | Non |
| `HEARTBEAT.md` | Tâches proactives (cron) | Non |

---

## 🚀 Quickstart

### Prérequis

- **Python 3.11+**
- **Ollama** installé et lancé → [ollama.com](https://ollama.com)

```bash
# Option A : Gemma 4
ollama pull gemma4          # 4B — minimum pour démarrer
ollama pull gemma4:26b      # 26B MoE — recommandé
ollama pull gemma4:31b      # 31B Dense — optionnel, pour le Critic

# Option B : Mistral
ollama pull mistral-small3.2
ollama pull ministral-3:8b

# Embeddings (requis pour la mémoire)
ollama pull nomic-embed-text
```

### Installation

```bash
git clone https://github.com/rapidoo/crabot.git && cd crabot
make install

# Configure tes secrets (optionnel)
cp .env.example .env
# Édite .env : TELEGRAM_BOT_TOKEN, NEO4J_PASSWORD, BRAVE_API_KEY
```

### 3 modes d'utilisation

```bash
# 💬 Mode interactif
make chat

# ⚡ Mode one-shot
make run PROMPT="Calcule pi avec 10 décimales"

# 🤖 Mode daemon — bot Telegram 24/7 (supervisor + worker)
make daemon
```

---

## 🤖 Mode Telegram (24/7)

```bash
# En service macOS (auto-start au boot, auto-restart on crash)
make service-install && make service-start
```

- Pipeline complet Plan → Execute → Critique sur chaque message
- **Heartbeat** toutes les 30 min — tâches proactives
- **Goal engine** toutes les 15 min — objectifs long terme
- **Apprentissage** — `/bad "raison"` extrait une lesson automatiquement
- Commandes : `/good`, `/bad`, `/stats`, `/evolve`, `/clean`, `/help`

---

## ⚙️ Configuration

Tout dans `agent/config.yaml`. Secrets dans `.env`.

```env
# .env — rien n'est obligatoire sauf Ollama
TELEGRAM_BOT_TOKEN=ton_token_botfather     # optionnel — mode Telegram
NEO4J_PASSWORD=ton_password_neo4j          # optionnel — mémoire persistante
BRAVE_API_KEY=ta_cle_brave_search          # optionnel — recherche web
MODEL_NAME=GEMMA4                          # ou MISTRAL
```

> **Le minimum pour démarrer : Ollama + un modèle. C'est tout.** Le reste s'active progressivement.

---

## 📁 Structure du projet

```
crabot/
├── workspace/                 🦀 Personnalité (évolue automatiquement)
│   ├── SOUL.md                    Ton, humour, opinions
│   ├── IDENTITY.md                Nom, emoji, vibe
│   ├── AGENTS.md                  Règles opérationnelles
│   ├── USER.md                    Profil utilisateur
│   └── HEARTBEAT.md               Tâches proactives
├── agent/
│   ├── agent.py               Orchestrateur principal
│   ├── supervisor.py          Supervisor (gère le worker)
│   ├── worker.py              Worker d'exécution (IPC)
│   ├── worker_evolution.py    Worker d'évolution (validation code)
│   ├── ipc.py                 Protocole IPC Unix socket
│   ├── config.yaml            Configuration modèles
│   ├── core/
│   │   ├── triage.py          Classification simple/complex
│   │   ├── planner.py         Plan JSON structuré
│   │   ├── executor.py        Exécution (parallel, loop detection)
│   │   ├── critic.py          Scoring + retry adaptatif
│   │   ├── scheduler.py       Cron + self-scheduling
│   │   └── reflection.py      Réflexion sur métriques
│   ├── memory/
│   │   ├── neo4j_client.py    Graphe (épisodes, entités, skills, lessons)
│   │   ├── embedder.py        Embeddings nomic-embed-text
│   │   ├── entity_extractor.py Extraction entités via LLM
│   │   └── lesson_extractor.py Extraction lessons (corrections user)
│   ├── intelligence/
│   │   ├── workspace_evolver.py  Évolution workspace depuis lessons
│   │   ├── lesson_injector.py    Injection lessons dans Planner
│   │   ├── skill_injector.py     Injection skills apprises
│   │   ├── prompt_manager.py     A/B testing de prompts
│   │   ├── action_applier.py     Self-modification (code, config)
│   │   └── loop_detection.py     3 détecteurs de boucles
│   ├── tools/
│   │   ├── registry.py        Auto-discovery
│   │   ├── code_exec.py       Sandbox Python
│   │   ├── web_search.py      Brave Search API
│   │   ├── tool_create.py     Meta-tool : auto-création
│   │   └── custom/            🔧 Outils générés par l'agent
│   └── interfaces/
│       └── telegram_bot.py    Bot Telegram
├── tests/
└── benchmarks/
```

---

## 🗺️ Roadmap

### Fait
- [x] Recherche hybride par embeddings (vector 70% + lexical 30%)
- [x] Conversation multi-tours (contexte glissant, 20 derniers échanges)
- [x] Compression mémorielle automatique
- [x] Apprentissage par lessons (corrections user persistées)
- [x] Architecture supervisor + worker (self-modification avec reload)
- [x] Évolution autonome du workspace (USER.md, AGENTS.md, SOUL.md)
- [x] Support multi-modèles (Gemma 4 + Mistral)
- [x] A/B testing de prompts avec promotion automatique

### En cours
- [ ] RAG sur documents locaux (PDF, Markdown, code source)
- [ ] Interface web locale (dashboard, métriques, configuration)
- [ ] Planning long terme (sous-goals avec suivi automatique)

### Prévu
- [ ] Docker Compose (Ollama + Neo4j + Crabot)
- [ ] Multi-utilisateurs (contextes isolés, rôles, quotas)
- [ ] API REST + MCP server
- [ ] Fine-tuning local (LoRA sur épisodes à score élevé)

---

## 🧪 Tests

```bash
make test                 # Unit + intégration
make test-unit            # Unit seuls — pas besoin d'Ollama
make test-integration     # Avec Ollama réel
make test-neo4j           # Mémoire graph
make benchmark            # 20 prompts avec métriques
```

---

## 📋 Commandes

```bash
make install              # Crée venv + installe tout
make chat                 # REPL interactif
make run PROMPT="..."     # One-shot
make daemon               # Bot Telegram (supervisor + worker)
make service-install      # Service macOS (LaunchAgent)
make service-start        # Démarre le service
make service-stop         # Arrête
make service-restart      # Redémarre
make service-logs         # Tail -f des logs
make setup-neo4j          # Neo4j via Docker
make clean                # Supprime venv, caches, logs
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Projet open source, né en Bretagne, ouvert au monde.

1. Fork le repo
2. Crée une branche (`git checkout -b feature/mon-truc`)
3. Commite (`git commit -m 'Add mon truc'`)
4. Push (`git push origin feature/mon-truc`)
5. Ouvre une Pull Request

---

## 📜 Licence

Apache 2.0 — libre d'utilisation, modification et distribution.

---

<div align="center">

**Crafté avec obstination en Bretagne 🦀🌊**<br>
<sub>Par <a href="https://github.com/flebris">Frédéric Le Bris</a> — Le Bris Consulting</sub><br>
<sub>Propulsé par <a href="https://ollama.com">Ollama</a>, <a href="https://ai.google.dev/gemma">Gemma 4</a> et <a href="https://mistral.ai">Mistral</a></sub>

*Aucun crabe n'a été maltraité pendant le développement de ce projet.*

</div>
