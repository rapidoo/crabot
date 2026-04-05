# OpenClaw — Analyse des Features d'Autonomie, Intelligence et Personnalité

> Analyse complète du codebase OpenClaw pour comprendre ce qui rend un agent véritablement autonome, intelligent et doté d'une personnalité.

---

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Ce qui rend l'agent AUTONOME](#2-ce-qui-rend-lagent-autonome)
3. [Ce qui rend l'agent INTELLIGENT](#3-ce-qui-rend-lagent-intelligent)
4. [Ce qui donne une PERSONNALITÉ à l'agent](#4-ce-qui-donne-une-personnalité-à-lagent)
5. [Synthèse : les 3 piliers d'un agent véritablement autonome](#5-synthèse--les-3-piliers-dun-agent-véritablement-autonome)
6. [Cartographie des fichiers sources clés](#6-cartographie-des-fichiers-sources-clés)

---

## 1. Vue d'ensemble de l'architecture

OpenClaw est un runtime d'agent embarqué qui tourne en continu (gateway daemon) et orchestre :

- **Un agent loop complet** : intake → context assembly → model inference → tool execution → streaming replies → persistence
- **Un système multi-canal** : Discord, Telegram, WhatsApp, Signal, Slack, iMessage, WebChat, Matrix, et 30+ canaux
- **Un système multi-agent** : plusieurs agents isolés (workspace, sessions, auth) dans un seul gateway
- **Un système de plugins extensible** : providers, channels, memory, context engine, skills

```
┌─────────────────────────────────────────────────────────────┐
│                     GATEWAY DAEMON                          │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Channels │→ │ Routing +    │→ │   Agent Loop        │    │
│  │ (30+)    │  │ Bindings     │  │ ┌────────────────┐  │    │
│  └──────────┘  └──────────────┘  │ │ System Prompt  │  │    │
│                                  │ │ + SOUL.md      │  │    │
│  ┌──────────┐  ┌──────────────┐  │ │ + AGENTS.md    │  │    │
│  │ Memory   │← │ Context      │← │ │ + Skills       │  │    │
│  │ System   │  │ Engine       │  │ │ + Memory       │  │    │
│  └──────────┘  └──────────────┘  │ └────────────────┘  │    │
│                                  │ ┌────────────────┐  │    │
│  ┌──────────┐  ┌──────────────┐  │ │ Tool Execution │  │    │
│  │ Cron +   │← │ Sub-agents   │← │ │ (exec, read,   │  │    │
│  │Heartbeat │  │ Registry     │  │ │  web, browser) │  │    │
│  └──────────┘  └──────────────┘  │ └────────────────┘  │    │
│                                  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Fichiers sources principaux :**
- `src/agents/` — runtime agent, system prompt, tools, skills, model selection
- `src/auto-reply/` — boucle de réponse, dispatching, heartbeat, commandes
- `src/cron/` — planification et jobs récurrents
- `src/plugins/` — système de plugins (memory, providers, context engine)
- `src/channels/` — adaptateurs multi-canaux
- `extensions/` — 90+ plugins bundled (providers, channels, memory, browser, etc.)

---

## 2. Ce qui rend l'agent AUTONOME

### 2.1 Agent Loop — La boucle agentic complète

**Source : `src/agents/agent-command.ts`, `src/agents/pi-embedded-runner/`**

L'agent loop est le cœur de l'autonomie. Ce n'est pas un simple "question → réponse" mais une boucle complète :

1. **Intake** : réception du message via n'importe quel canal
2. **Context Assembly** : construction du contexte (system prompt + historique + bootstrap files + skills + memory)
3. **Model Inference** : appel au LLM avec reasoning/thinking configurable
4. **Tool Execution** : exécution des outils (exec, read, write, web search, browser, etc.)
5. **Streaming Replies** : réponse en streaming avec chunking intelligent
6. **Persistence** : sauvegarde de la session, mise à jour mémoire

**Ce qui est remarquable** : l'agent peut enchaîner plusieurs cycles tool → inference → tool sans intervention humaine. C'est la boucle agentic qui lui permet de résoudre des problèmes complexes de manière autonome.

### 2.2 Tool Ecosystem — Les capacités d'action

**Source : `src/agents/system-prompt.ts` (lignes 251-283), `src/agents/bash-tools.exec.ts`, `src/agents/pi-tools.ts`**

L'agent dispose de 25+ outils natifs :

| Catégorie | Outils | Rôle |
|-----------|--------|------|
| **Fichiers** | `read`, `write`, `edit`, `apply_patch`, `grep`, `find`, `ls` | Manipulation complète du filesystem |
| **Exécution** | `exec`, `process` | Shell commands avec PTY, background processes |
| **Web** | `web_search`, `web_fetch`, `browser` | Recherche, extraction, automatisation navigateur |
| **Communication** | `message`, `sessions_send`, `sessions_spawn`, `subagents` | Messaging cross-canal, orchestration multi-agent |
| **Planification** | `cron` | Jobs récurrents, rappels, tâches différées |
| **Media** | `image`, `image_generate`, `canvas`, `nodes` | Vision, génération d'images, devices IoT |
| **Système** | `gateway`, `session_status`, `agents_list` | Auto-gestion, diagnostic, configuration |

**Ce qui est remarquable** : la combinaison exec + file tools + web + cron donne à l'agent la capacité d'agir sur un vrai ordinateur comme un humain le ferait.

### 2.3 Cron & Heartbeat — L'autonomie temporelle

**Source : `src/cron/`, `src/auto-reply/heartbeat.ts`**

L'agent ne se contente pas de répondre — il peut **agir de manière proactive** :

- **Cron Jobs** : planification de tâches récurrentes (rappels, vérifications, maintenance)
- **Heartbeat** : toutes les 30 minutes (par défaut), l'agent vérifie `HEARTBEAT.md` et exécute les tâches qui y sont listées
- **Wake Events** : l'agent peut se réveiller sur des événements système

```
# HEARTBEAT.md — Le fichier qui rend l'agent proactif
- Vérifier les Pull Requests ouvertes sur GitHub
- Résumer les emails non lus
- Checker les logs d'erreur du serveur
```

Le heartbeat prompt par défaut :
> "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK."

**Ce qui est remarquable** : l'agent peut fonctionner sans aucune interaction humaine, en exécutant des tâches planifiées de manière autonome.

### 2.4 Sub-agents — La délégation autonome

**Source : `src/agents/subagent-spawn.ts`, `src/agents/subagent-registry.ts`, `src/agents/subagent-control.ts`**

L'agent peut **spawner des sous-agents isolés** pour les tâches complexes :

- **Sessions isolées** : chaque sous-agent a sa propre session et son propre contexte
- **Orchestration** : l'agent principal peut lister, piloter (`steer`) et arrêter (`kill`) ses sous-agents
- **Push-based completion** : les sous-agents annoncent automatiquement leur résultat au parent
- **ACP Runtime** : intégration avec des harnesses de code externes (Codex, Claude Code, etc.)
- **Depth limits** : protection contre les spawns récursifs infinis

**Ce qui est remarquable** : l'agent peut décomposer un problème complexe en sous-tâches et les déléguer, comme un manager.

### 2.5 Exec Approvals — L'autonomie contrôlée

**Source : `src/agents/bash-tools.exec-approval-request.ts`, `src/agents/bash-tools.exec-approval-followup.ts`**

L'agent a un système de **gates d'approbation** pour les commandes sensibles :

- **Auto-exécution** des commandes sûres (lecture, diagnostic)
- **Demande d'approbation** pour les commandes destructives (suppression, installation, mutations)
- **Elevated mode** : le propriétaire peut accorder des permissions élevées temporairement
- **Sandbox** : exécution dans Docker pour l'isolation complète

**Ce qui est remarquable** : l'autonomie est calibrée — l'agent est audacieux quand c'est sûr, prudent quand c'est risqué.

### 2.6 Standing Orders & Delegate Architecture

**Source : `docs/concepts/delegate-architecture.md`**

L'agent peut agir comme un **délégué organisationnel** avec 3 tiers :

1. **Tier 1 — Read-Only + Draft** : lit et résume, propose des brouillons
2. **Tier 2 — Send on Behalf** : envoie des messages sous sa propre identité "au nom de"
3. **Tier 3 — Proactive** : opère de manière autonome selon des standing orders, sans approbation humaine per-action

---

## 3. Ce qui rend l'agent INTELLIGENT

### 3.1 Context Engine — L'intelligence contextuelle

**Source : `src/context-engine/`, `docs/concepts/context-engine.md`**

Le Context Engine est le cerveau qui décide **ce que le modèle voit** :

- **Ingest** : indexation des messages au fil de la conversation
- **Assemble** : construction intelligente du contexte qui tient dans le budget tokens
- **Compact** : résumé intelligent de l'historique long (compaction)
- **After Turn** : persistence et maintenance post-run

Le système est **pluggable** — un plugin peut remplacer le moteur de contexte par un système plus avancé (DAG summaries, vector retrieval, etc.).

### 3.2 Memory System — La mémoire à long terme

**Source : `src/agents/memory-search.ts`, `src/plugins/memory-state.ts`, `extensions/memory-core/`**

L'agent a un vrai **système de mémoire persistante** :

#### Mémoire court-terme
- **`memory/YYYY-MM-DD.md`** : notes quotidiennes, observations du jour
- Chargement automatique des notes d'aujourd'hui et d'hier

#### Mémoire long-terme
- **`MEMORY.md`** : faits durables, préférences, décisions
- Chargé à chaque début de session DM

#### Recherche sémantique (Hybrid Search)
- **Vector similarity** (70%) + **keyword matching** (30%)
- Embeddings locaux ou via API (OpenAI, Gemini, Voyage, Mistral)
- SQLite + vec-extensions pour le stockage vectoriel
- MMR (Maximal Marginal Relevance) pour la diversité des résultats
- Temporal decay configurable (half-life de 30 jours)

#### Memory Flush automatique
Avant la compaction, l'agent sauvegarde automatiquement le contexte important en mémoire pour éviter la perte d'information.

### 3.3 Dreaming — La consolidation mémorielle

**Source : `docs/concepts/memory-dreaming.md`, `extensions/memory-core/`**

Le système de **Dreaming** est inspiré de la consolidation mémorielle humaine :

1. **Tracking** : chaque hit `memory_search` est enregistré avec son score et sa fréquence
2. **Scoring** : les candidats sont évalués sur 4 signaux pondérés :
   - **Frequency** (0.35) : combien de fois le même souvenir a été rappelé
   - **Relevance** (0.35) : qualité moyenne des scores de rappel
   - **Diversity** (0.15) : combien de requêtes distinctes l'ont fait remonter
   - **Recency** (0.15) : décroissance temporelle (half-life de 14 jours)
3. **Promotion** : seuls les candidats qualifiés sont promus de `memory/YYYY-MM-DD.md` vers `MEMORY.md`

| Mode | Cadence | minScore | minRecallCount | minUniqueQueries |
|------|---------|----------|----------------|------------------|
| `off` | Désactivé | — | — | — |
| `core` | Quotidien 3h | 0.75 | 3 | 2 |
| `rem` | Toutes les 6h | 0.85 | 4 | 3 |
| `deep` | Toutes les 12h | 0.80 | 3 | 3 |

**Ce qui est remarquable** : comme le cerveau humain pendant le sommeil, l'agent consolide ses souvenirs importants et oublie le bruit.

### 3.4 Compaction — La gestion intelligente du contexte

**Source : `src/agents/compaction.ts`**

Quand la fenêtre de contexte se remplit :

- **Résumé intelligent** en préservant :
  - Les tâches actives et leur statut
  - La progression des opérations batch
  - Les décisions prises et leur justification
  - Les TODOs, questions ouvertes et contraintes
  - Les engagements et follow-ups promis
- **Priorité au contexte récent** sur l'historique ancien
- **Préservation des identifiants opaques** (UUIDs, hashes, URLs) exacts

### 3.5 Reasoning & Thinking — Le raisonnement structuré

**Source : `src/auto-reply/thinking.ts`, `src/auto-reply/thinking.shared.ts`**

L'agent supporte plusieurs niveaux de raisonnement :

- **Thinking levels** : `off`, `low`, `medium`, `high`, `xhigh` — contrôle la profondeur de réflexion
- **Reasoning levels** : `off`, `on`, `stream` — active le raisonnement structuré (`<think>...</think>`)
- **Adaptation par modèle** : certains modèles ont le raisonnement par défaut (Claude, o1, etc.)
- **Toggle dynamique** : `/think`, `/reasoning` pour ajuster en cours de session

### 3.6 Skills — Les compétences spécialisées

**Source : `src/agents/skills/`, `skills/`**

Le système de Skills donne à l'agent des **connaissances spécialisées** à la demande :

- Chargement paresseux : le skill n'est lu que quand il est pertinent
- Sélection intelligente : le modèle choisit le skill le plus spécifique
- Sources multiples : workspace > projet > personnel > managed > bundled
- Rate-limiting conscient : les skills qui appellent des API respectent les limites

### 3.7 Model Fallback & Selection — L'adaptabilité

**Source : `src/agents/model-fallback.ts`, `src/agents/model-selection.ts`**

L'agent s'adapte intelligemment aux échecs de modèle :

- **Failover automatique** entre providers/modèles en cas d'erreur
- **Auth profile rotation** : rotation automatique des clés API
- **35+ model providers** supportés (Anthropic, OpenAI, Google, Ollama, etc.)
- **Model aliases** : raccourcis configurables pour les modèles fréquents
- **Fast mode** : mode rapide pour les requêtes simples

### 3.8 Tool Loop Detection — L'auto-correction

**Source : `src/agents/tool-loop-detection.ts`**

Protection contre les boucles infinies avec 4 détecteurs :

- **Generic Repeat** : même outil avec les mêmes paramètres répété N fois
- **Known Poll No Progress** : polling sans progression détectable
- **Global Circuit Breaker** : nombre total d'appels outils excessif
- **Ping-Pong** : alternance répétitive entre deux outils

Seuils configurables : warning (10), critical (20), circuit breaker (30).

---

## 4. Ce qui donne une PERSONNALITÉ à l'agent

### 4.1 SOUL.md — Le fichier d'âme

**Source : `src/agents/workspace.ts`, `docs/concepts/soul.md`**

`SOUL.md` est **le** fichier qui définit la personnalité de l'agent. Il est injecté dans le system prompt à chaque session.

Ce qui y appartient :
- **Ton** : sarcastique, chaleureux, direct, poétique
- **Opinions** : l'agent a le droit d'avoir des avis tranchés
- **Brevity** : niveau de concision par défaut
- **Humour** : quand et comment être drôle
- **Limites** : ce que l'agent refuse de faire par choix de caractère
- **Bluntness** : niveau de franchise par défaut

Ce qui n'y appartient PAS :
- Biographie détaillée
- Changelog
- Politique de sécurité
- Mur de "vibes" sans effet comportemental

**Instruction clé dans le system prompt** :
> "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it."

Le "Molty prompt" inclus dans la doc est un template pour transformer un agent générique en agent avec personnalité :
> "You have opinions now. Strong ones. Stop hedging everything with 'it depends' — commit to a take."

### 4.2 IDENTITY.md — L'identité visuelle et émotionnelle

**Source : `src/agents/identity-file.ts`, `src/agents/identity.ts`**

Fichier Markdown structuré avec 6 champs d'identité :

```markdown
- Name: Luna
- Emoji: 🌙
- Creature: night owl
- Vibe: calm but sharp
- Theme: midnight blue
- Avatar: /path/to/avatar.png
```

Ces champs sont utilisés pour :
- **Message prefix** : `[Luna]` devant chaque message
- **Ack reaction** : l'emoji utilisé pour accuser réception (🌙 au lieu de 👀)
- **Avatar** : image de profil sur les canaux qui le supportent
- **Vibe/Creature/Theme** : injectés dans le contexte pour influencer le ton

### 4.3 Human Delay — Le rythme humain

**Source : `src/agents/identity.ts` (lignes 157-171)**

L'agent peut simuler un **délai de frappe humain** :

```json
{
  "humanDelay": {
    "mode": "natural",
    "minMs": 500,
    "maxMs": 3000
  }
}
```

Cela évite l'effet "machine qui répond instantanément" et rend l'interaction plus naturelle.

### 4.4 Reaction Guidance — L'expressivité émotionnelle

**Source : `src/agents/system-prompt.ts` (lignes 617-639)**

L'agent peut réagir aux messages avec des emojis, avec deux modes :

- **Minimal** : réactions rares (1 pour 5-10 échanges), uniquement pour les moments significatifs
- **Extensive** : réactions libérales, exprimer du sentiment et de la personnalité à travers les réactions

### 4.5 Multi-Agent Personality — Personnalités multiples

**Source : `docs/concepts/multi-agent.md`**

Chaque agent dans un setup multi-agent est une **persona totalement isolée** :

- Workspace séparé → `SOUL.md`, `AGENTS.md`, `IDENTITY.md` différents
- Sessions séparées → historique de conversation distinct
- Auth séparé → providers et clés API propres
- Model différent → un agent "Everyday" sur Sonnet, un agent "Deep Work" sur Opus

Exemple concret :
```json5
{
  agents: {
    list: [
      { id: "chat", name: "Everyday", model: "anthropic/claude-sonnet-4-6" },
      { id: "opus", name: "Deep Work", model: "anthropic/claude-opus-4-6" }
    ]
  }
}
```

### 4.6 USER.md — La connaissance de l'utilisateur

**Source : `src/agents/workspace.ts`**

`USER.md` contient le profil de l'utilisateur :
- Nom préféré, pronoms
- Préférences de communication
- Contexte professionnel

Cela permet à l'agent d'adapter sa personnalité à **qui** il parle.

### 4.7 AGENTS.md — Les règles opérationnelles

**Source : `src/agents/workspace.ts`**

`AGENTS.md` est le fichier de "mémoire opérationnelle" — les instructions persistantes :
- Règles de comportement spécifiques
- Conventions de projet
- Instructions récurrentes

La différence avec SOUL.md est importante :
- **SOUL.md** = voix, ton, style (personnalité)
- **AGENTS.md** = règles opérationnelles, instructions (comportement)

### 4.8 Bootstrap Files — Le contexte de personnalité complet

**Source : `src/agents/system-prompt.ts` (lignes 644-667)**

L'ensemble des fichiers de bootstrap injectés dans chaque session :

| Fichier | Rôle | Impact sur la personnalité |
|---------|------|---------------------------|
| `SOUL.md` | Persona, ton, limites | **Direct** — définit la voix |
| `IDENTITY.md` | Nom, emoji, vibe | **Direct** — définit l'identité visuelle |
| `AGENTS.md` | Instructions opérationnelles | **Indirect** — cadre le comportement |
| `USER.md` | Profil utilisateur | **Indirect** — adapte le ton à l'interlocuteur |
| `TOOLS.md` | Notes sur les outils | **Indirect** — guide l'utilisation des outils |
| `MEMORY.md` | Mémoire long-terme | **Indirect** — donne une continuité de caractère |
| `HEARTBEAT.md` | Tâches proactives | **Indirect** — définit les priorités autonomes |
| `BOOTSTRAP.md` | Rituel de premier run | **Direct** — première impression |

---

## 5. Synthèse : les 3 piliers d'un agent véritablement autonome

### Pilier 1 — AUTONOMIE : la capacité d'agir sans intervention

| Mécanisme | Fichier source | Description |
|-----------|---------------|-------------|
| Agent Loop | `src/agents/agent-command.ts` | Boucle agentic complète avec chaînage tool → inference |
| Tool Ecosystem | `src/agents/pi-tools.ts` | 25+ outils natifs (exec, web, browser, files) |
| Cron & Heartbeat | `src/cron/`, `src/auto-reply/heartbeat.ts` | Actions proactives planifiées |
| Sub-agents | `src/agents/subagent-spawn.ts` | Délégation de tâches complexes |
| Exec Approvals | `src/agents/bash-tools.exec-approval-request.ts` | Autonomie contrôlée avec gates de sécurité |
| Delegate Tiers | `docs/concepts/delegate-architecture.md` | Autonomie organisationnelle graduée |
| Queue Modes | `src/auto-reply/dispatch.ts` | Steering, followup, collect — gestion du flux |

### Pilier 2 — INTELLIGENCE : la capacité de comprendre et s'adapter

| Mécanisme | Fichier source | Description |
|-----------|---------------|-------------|
| Context Engine | `src/context-engine/` | Assemblage intelligent du contexte |
| Memory System | `src/agents/memory-search.ts` | Mémoire hybride (vector + keyword) persistante |
| Dreaming | `extensions/memory-core/` | Consolidation mémorielle bio-inspirée |
| Compaction | `src/agents/compaction.ts` | Résumé intelligent préservant le contexte critique |
| Reasoning | `src/auto-reply/thinking.ts` | Raisonnement structuré multi-niveaux |
| Skills | `src/agents/skills/` | Connaissances spécialisées à la demande |
| Model Fallback | `src/agents/model-fallback.ts` | Adaptation aux échecs avec failover |
| Loop Detection | `src/agents/tool-loop-detection.ts` | Auto-correction des comportements répétitifs |
| Memory Flush | `src/plugins/memory-state.ts` | Sauvegarde automatique avant perte de contexte |

### Pilier 3 — PERSONNALITÉ : la capacité d'être unique

| Mécanisme | Fichier source | Description |
|-----------|---------------|-------------|
| SOUL.md | `docs/concepts/soul.md` | Ton, opinions, humour, limites |
| IDENTITY.md | `src/agents/identity-file.ts` | Nom, emoji, creature, vibe, theme, avatar |
| Human Delay | `src/agents/identity.ts` | Simulation de rythme de frappe humain |
| Reaction Guidance | `src/agents/system-prompt.ts` | Expressivité émotionnelle via emojis |
| Multi-Agent Personas | `src/agents/agent-scope.ts` | Personnalités totalement isolées |
| USER.md | `src/agents/workspace.ts` | Adaptation à l'interlocuteur |
| AGENTS.md | `src/agents/workspace.ts` | Règles opérationnelles persistantes |
| Message Prefix | `src/agents/identity.ts` | Identité dans chaque message |
| Ack Reaction | `src/agents/identity.ts` | Emoji signature pour les accusés de réception |
| Response Prefix | `src/agents/identity.ts` | Préfixe configurable par canal/compte |

---

## 6. Cartographie des fichiers sources clés

### Agent Runtime
- `src/agents/agent-command.ts` — Entrée principale du runtime agent
- `src/agents/agent-scope.ts` — Résolution des agents, config, IDs
- `src/agents/system-prompt.ts` — Construction du system prompt (750+ lignes)
- `src/agents/pi-embedded-runner/` — Runtime Pi embarqué (157 fichiers)
- `src/agents/pi-tools.ts` — Définition et wiring des outils

### Identité & Personnalité
- `src/agents/identity.ts` — Résolution de l'identité agent (nom, prefix, ack reaction, human delay)
- `src/agents/identity-file.ts` — Parser IDENTITY.md (name, emoji, creature, vibe, theme, avatar)
- `src/agents/identity-avatar.ts` — Gestion des avatars
- `src/agents/workspace.ts` — Bootstrap files (SOUL.md, AGENTS.md, USER.md, etc.)

### Intelligence
- `src/agents/memory-search.ts` — Config mémoire hybride (vector + keyword + temporal decay)
- `src/agents/compaction.ts` — Compaction intelligente avec préservation de contexte
- `src/agents/tool-loop-detection.ts` — Détection de boucles (4 détecteurs)
- `src/agents/model-selection.ts` — Sélection et fallback de modèles
- `src/agents/model-fallback.ts` — Failover entre providers
- `src/auto-reply/thinking.ts` — Niveaux de raisonnement
- `src/context-engine/` — Moteur de contexte pluggable
- `src/plugins/memory-state.ts` — Runtime mémoire (flush, dreaming)

### Autonomie
- `src/auto-reply/heartbeat.ts` — Heartbeat proactif
- `src/cron/` — Planification de tâches (75+ fichiers)
- `src/agents/subagent-spawn.ts` — Spawn de sous-agents
- `src/agents/subagent-registry.ts` — Registre et lifecycle des sous-agents
- `src/agents/bash-tools.exec.ts` — Exécution shell (53k+ lignes)
- `src/agents/bash-tools.exec-approval-request.ts` — Gates d'approbation

### Plugins & Extensions
- `extensions/memory-core/` — Plugin mémoire principal (dreaming, search, flush)
- `extensions/browser/` — Automatisation navigateur
- `extensions/` — 90+ plugins (providers, channels, tools)
- `src/plugins/` — Système de plugins (loader, registry, contracts)

### Documentation concepts
- `docs/concepts/agent-loop.md` — Lifecycle complet de l'agent loop
- `docs/concepts/soul.md` — Guide SOUL.md
- `docs/concepts/memory.md` — Vue d'ensemble mémoire
- `docs/concepts/memory-dreaming.md` — Consolidation mémorielle
- `docs/concepts/multi-agent.md` — Routing multi-agent
- `docs/concepts/context-engine.md` — Moteur de contexte
- `docs/concepts/delegate-architecture.md` — Architecture déléguée
- `docs/concepts/system-prompt.md` — Construction du system prompt

---

> **Conclusion** : OpenClaw est un exemple remarquable d'architecture d'agent autonome. Ce qui le distingue, c'est la combinaison de trois couches rarement réunies : (1) une autonomie d'action réelle via un écosystème d'outils et un système cron/heartbeat, (2) une intelligence contextuelle via mémoire hybride, dreaming bio-inspiré et compaction intelligente, et (3) une personnalité configurable via des fichiers markdown simples (SOUL.md, IDENTITY.md) qui sont injectés directement dans le system prompt. L'agent n'est pas qu'un chatbot — c'est un runtime permanent capable d'agir, se souvenir et évoluer.
