# Pattern Agentique : Plan → Execute → Critique
**Compensation d'un modèle local faible par un scaffold fort**

*Le Bris Consulting — Avril 2025 — v1.1*
*Contexte : Gemma 4 (E4B / 26B MoE / 31B Dense) · MacBook M4 Pro 48GB · Ollama*

---

## 1. Contexte et objectif

Les modèles LLM locaux de taille intermédiaire (Gemma 4 26B MoE, Qwen3-Coder, Mistral Small)
offrent souveraineté des données et gratuité à l'inférence, mais restent en retrait des APIs
cloud sur les tâches nécessitant un raisonnement complexe multi-étapes.

Ce document décrit un pattern agentique permettant de compenser cette faiblesse par un
**scaffold fort** : une architecture de contrôle qui décompose, vérifie et itère plutôt que de
demander au modèle de tout résoudre en un seul appel.

> *Un modèle faible avec un scaffold fort bat souvent un modèle fort sans scaffold,
> sur les tâches structurables.*

### Hypothèses de départ

- **Zéro dépendance cloud** : agent 100% local, pas de SDK tiers, appels directs Ollama API
- Hardware : Apple Silicon M4 Pro, 48 GB unifié, Ollama (llama.cpp backend)
- Multi-modèle : mixer les variantes Gemma 4 selon le besoin (voir ci-dessous)
- Contexte disponible : 128K–256K tokens selon variante

### Famille Gemma 4 — variantes disponibles

| Variante | Params totaux | Actifs | Contexte | Modalités | RAM (Q4) | Rôle dans l'agent |
|----------|--------------|--------|----------|-----------|----------|-------------------|
| **E2B** | 5.1B (2.3B eff) | 2.3B | 128K | Text, Image, Audio | 7.2 GB | Triage ultra-rapide |
| **E4B** | 8B (4.5B eff) | 4.5B | 128K | Text, Image, Audio | 9.6 GB | Draft, classification |
| **26B MoE** | 25.2B | 3.8B | 256K | Text, Image | 18 GB | Backbone (Plan + Execute) |
| **31B Dense** | 30.7B | 30.7B | 256K | Text, Image | 20 GB | Critic fort |

Architecture MoE 26B : 128 experts totaux, 8 actifs + 1 shared par token.
Licence Apache 2.0. Function calling natif (entraîné, pas instruction-tuned).
Thinking mode natif via token `<|think|>`. System prompt natif (rôle `system`).

**Contrainte RAM** : 26B (18GB) + E4B (9.6GB) = 27.6 GB en simultané. Le 31B Dense
(20GB) nécessite de décharger le 26B. Ollama gère le swap modèle automatiquement mais
avec une pénalité de ~5s au chargement. Stratégie : garder 26B résident, charger 31B
uniquement pour les passes Critic à fort enjeu.

---

## 2. Architecture du pattern

### 2.1 Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────┐
│                         USER INPUT                             │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                        ↓  [MEMORY NODE]  ↓
                   enrichissement depuis Neo4j
                               │
          ┌────────────────────┴───────────────────┐
          │         PHASE 1 : PLANNER              │
          │  • Analyse de la tâche                 │
          │  • Sortie : JSON structuré             │
          │  • Steps + outils requis               │
          └────────────────────┬───────────────────┘
                               │ plan.json
          ┌────────────────────┴───────────────────┐
          │         PHASE 2 : EXECUTOR             │
          │  • Execution séquentielle / //         │
          │  • Tool calls natifs Gemma 4           │
          │  • Skeleton-of-Thought si long         │
          └────────────────────┬───────────────────┘
                               │ results[]
          ┌────────────────────┴───────────────────┐
          │         PHASE 3 : CRITIC               │
          │  • Score 0-10 par step                 │
          │  • Retry si score < seuil              │
          │  • Mise à jour mémoire Neo4j           │
          └────────────────────┬───────────────────┘
                               │
                           OUTPUT FINAL
                               │
                        [MEMORY WRITE]
                    persistance Neo4j
```

### 2.2 Détail des phases

| Phase | Composant | Rôle | Output attendu |
|---|---|---|---|
| Memory read | Neo4j | Récupère le contexte pertinent (entités, épisodes passés) | Contexte enrichi injecté dans le prompt du Planner |
| Plan | Planner LLM | Analyse la tâche, émet un plan JSON structuré | `plan.json` : `[{step, tool, input, expected_output}]` |
| Execute | Executor LLM | Exécute chaque step du plan, appelle les outils | `results[]` : `[{step, output, tokens_used}]` |
| Critique | Critic LLM | Note chaque résultat, décide retry ou passage au suivant | `score[]`, `retry_flags[]`, `memory_updates[]` |
| Memory write | Neo4j | Persiste l'épisode, les entités clés, les scores | Noeuds Episode, Entity, Skill mis à jour |

---

## 3. Patterns de compensation du modèle

### 3.1 Plan-then-Execute

Le modèle est bien meilleur sur une tâche simple et définie que sur une tâche ouverte et
complexe. Forcer un plan structuré en sortie du Planner avant toute exécution réduit l'erreur
de dérive.

**Prompt système Planner :**

```
You are a task planner. Given a user request and memory context, produce
a JSON plan ONLY. No prose. No explanation.

Output format (strict):
{
  "goal": "one sentence description of the overall goal",
  "steps": [
    {
      "id": 1,
      "tool": "search|code|memory|file|none",
      "input": "exact input to pass to the tool or LLM",
      "expected_output": "what a correct result looks like"
    }
  ],
  "parallel": [1, 2]   // IDs of steps that can run concurrently
}
```

**Règles de validation du plan (pydantic) :**

```python
class Step(BaseModel):
    id: int
    tool: Literal["search", "code", "memory", "file", "none"]
    input: str
    expected_output: str

class Plan(BaseModel):
    goal: str
    steps: list[Step]
    parallel: list[int] = []
```

### 3.2 LLM-as-Judge (auto-critique)

Le Critic est la même instance Gemma 4 avec un system prompt antagoniste. Il évalue chaque
output de l'Executor sur une grille fixe et décide en autonomie du retry.

**Prompt système Critic :**

```
You are a strict quality evaluator. Score the following output against
the expected result. Return JSON ONLY. No prose.

Criteria (score 0-10 each):
  - completeness : does the output fully cover the expected_output?
  - accuracy     : no detectable factual hallucination?
  - format       : does the output match the requested format?
  - coherence    : consistent with previous steps context?

Output format (strict):
{
  "scores": {
    "completeness": X,
    "accuracy": X,
    "format": X,
    "coherence": X
  },
  "final_score": X.X,
  "retry": true|false,
  "reason": "one sentence explanation if retry=true"
}

Retry threshold: final_score < 6.5
Max retries per step: 3
```

### 3.3 Skeleton-of-Thought (SoT)

Sur les générations longues (> 500 tokens), le modèle dérive. Le SoT force une structure
squelette d'abord, puis le remplissage section par section.

**Séquence :**

1. **Appel 1 — Skeleton** : "Generate ONLY section titles and a one-sentence summary for each. No content."
2. **Appels 2..N — Fill** : Pour chaque section, appel indépendant avec uniquement le contexte nécessaire à cette section.
3. **Assemblage** : concaténation finale, vérification de cohérence par le Critic.

**Quand activer :** `len(expected_output_tokens) > config.skeleton_threshold` (défaut : 500)

**Avantage principal :** les sections indépendantes peuvent être parallélisées via `asyncio.gather`.

### 3.4 Routeur multi-modèle

Plutôt qu'un seul backbone, l'agent route chaque appel vers la variante optimale.
Le routeur est un triage E4B ultra-rapide en amont du pipeline.

```
                          USER INPUT
                              │
                    ┌─────────┴─────────┐
                    │  TRIAGE (E4B)     │  thinking off, < 1s
                    │  classify(input)  │
                    └────┬────┬────┬────┘
                         │    │    │
                simple   │    │    │  critique
                ◄────────┘    │    └────────►
                              │
                         complexe
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   E4B direct          26B MoE (Plan+Exec)     31B Dense (Critic)
   thinking off        thinking on              thinking on
   réponse courte      raisonnement struct.     évaluation rigoureuse
```

**Heuristiques de routage :**

| Condition | Modèle | Thinking |
|-----------|--------|----------|
| Input < 100 tokens, pas d'image, tâche simple | E4B | off |
| Tâche structurable (code, extraction, plan) | 26B MoE | on |
| Évaluation critique, décision importante | 31B Dense | on |
| Image / document OCR | 26B ou 31B (vision 550M) | on |
| Audio | E4B ou E2B (seuls avec audio encoder) | off |

**Speculative Draft** (cas particulier) : pour les steps Executor sans tool call,
E4B génère un draft rapide (~2s), le 26B vérifie et corrige (~5s). Gain net quand
le draft est accepté tel quel (~60% des cas sur les tâches simples).

```python
class ModelRouter:
    """Sélection du modèle optimal par step — inspiré du channel registry NanoClaw."""

    MODELS = {
        "triage":   "gemma4:e4b",
        "plan":     "gemma4:26b",
        "execute":  "gemma4:26b",
        "draft":    "gemma4:e4b",
        "critic":   "gemma4:31b",
        "critic_light": "gemma4:26b",  # fallback si RAM insuffisante
    }

    def select(self, step: Step, high_stakes: bool = False) -> str:
        if high_stakes:
            return self.MODELS["critic"]
        if step.tool == "none" and len(step.input) < 200:
            return self.MODELS["draft"]
        return self.MODELS["execute"]
```

### 3.5 Thinking Mode natif

Gemma 4 intègre un mode raisonnement activable par un token de contrôle dans le system prompt.
Le modèle sépare son raisonnement interne de sa réponse finale.

**Activation :** préfixer le system prompt avec `<|think|>`

**Output structuré :**
```
<|channel>thought
[raisonnement interne — non exposé à l'utilisateur]
<channel|>
[réponse finale]
```

**Stratégie par phase :**

| Phase | Thinking | Justification |
|-------|----------|---------------|
| Triage | off | Vitesse, classification simple |
| Planner | on | Décomposition = raisonnement structuré |
| Executor (tool call) | off | Exécution directe, pas d'ambiguïté |
| Executor (génération) | on | Génération longue = risque de dérive |
| Critic | on | Évaluation = raisonnement critique |

**Best practice Gemma 4 :** en multi-turn, ne jamais inclure les blocs `thought`
des tours précédents dans l'historique. Seule la réponse finale est conservée.

### 3.6 Sampling (recommandations Google)

Paramètres standardisés pour Gemma 4 :

```python
SAMPLING_DEFAULT  = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
SAMPLING_CRITIC   = {"temperature": 0.3, "top_p": 0.9,  "top_k": 32}   # déterministe
SAMPLING_PLANNER  = {"temperature": 0.7, "top_p": 0.95, "top_k": 64}   # structuré
```

### 3.7 Patterns d'infrastructure

Patterns extraits de l'analyse de [NanoClaw](https://github.com/qwibitai/nanoclaw),
un framework agent containerisé. Transposés ici pour un agent local autonome.

**Tool Registry (factory pattern)**

Découplage total entre l'orchestrateur et les outils disponibles. Chaque outil
s'enregistre lui-même ; l'orchestrateur ne connaît que l'interface.

```python
_registry: dict[str, Callable[[], Tool | None]] = {}

def register_tool(name: str, factory: Callable[[], Tool | None]) -> None:
    _registry[name] = factory

def get_tool(name: str) -> Tool | None:
    factory = _registry.get(name)
    return factory() if factory else None   # None = outil indisponible, skip
```

Ajouter un outil = créer un fichier dans `tools/`, appeler `register_tool` à l'import.
Zéro modification dans `agent.py` ou `executor.py`.

**Exécution sandboxée (subprocess isolé)**

Pour `code_exec`, isoler via subprocess avec timeout et restrictions :

```python
async def sandbox_exec(code: str, timeout: int = 30) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", code,
        stdout=PIPE, stderr=PIPE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        return ExecResult(code=proc.returncode, stdout=stdout, stderr=stderr)
    except asyncio.TimeoutError:
        proc.kill()
        return ExecResult(code=-1, stdout=b"", stderr=b"timeout")
```

**Cursor recovery (résilience aux crashs)**

Persister l'avancement du plan step par step. En cas de crash, reprendre
au dernier step réussi plutôt que depuis zéro.

```python
# Après chaque step réussi :
state.cursor = step.id
state.persist()  # SQLite ou JSON atomique (write tmp + rename)

# Au redémarrage :
plan = state.load_plan()
resume_from = state.cursor + 1  # reprendre au step suivant
```

**Retry exponentiel**

Sur les échecs Ollama (OOM, timeout, modèle pas chargé) :

```python
BASE_RETRY_MS = 2000
MAX_RETRIES = 4

async def with_retry(fn, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return await fn(*args)
        except OllamaError:
            await asyncio.sleep(BASE_RETRY_MS * 2**attempt / 1000)
    raise MaxRetriesExceeded()
```

---

### 3.8 Memory-Augmented Context (Neo4j)

Le graph Neo4j est l'avantage différenciant vs API cloud : une mémoire structurée et
persistante, sans coût marginal par session.

**Schéma Neo4j :**

```cypher
(:Episode {id, timestamp, summary, score, goal})
  -[:HAS_ENTITY]->  (:Entity  {name, type, description})
  -[:USED_SKILL]->  (:Skill   {name, success_rate, last_used})
  -[:FOLLOWS]->     (:Episode)

(:Entity)-[:RELATED_TO {weight}]->(:Entity)
```

**Requête d'enrichissement (Memory read au démarrage du Planner) :**

```cypher
// Top-K épisodes pertinents pour les entités détectées dans l'input
MATCH (e:Entity)<-[:HAS_ENTITY]-(ep:Episode)
WHERE e.name IN $detected_entities
  AND ep.score > 6.5
RETURN ep.summary, ep.timestamp, ep.goal
ORDER BY ep.timestamp DESC
LIMIT $context_k
```

**Requête de persistance (Memory write après Critic) :**

```cypher
// Créer l'épisode
MERGE (ep:Episode {id: $episode_id})
SET ep.timestamp = datetime(), ep.summary = $summary,
    ep.score = $final_score, ep.goal = $goal

// Lier les entités extraites
FOREACH (entity IN $entities |
  MERGE (e:Entity {name: entity.name})
  SET e.type = entity.type, e.description = entity.description
  MERGE (ep)-[:HAS_ENTITY]->(e)
)

// Chaîner avec l'épisode précédent
WITH ep
MATCH (prev:Episode {id: $previous_episode_id})
MERGE (ep)-[:FOLLOWS]->(prev)
```

---

## 4. Structure du projet

```
agent/
  ├── core/
  │   ├── planner.py        # LLM call + JSON schema validation (pydantic)
  │   ├── executor.py       # Tool dispatch + parallel execution (asyncio)
  │   └── critic.py         # Scoring + retry logic
  ├── memory/
  │   ├── neo4j_client.py   # Read/write graph, connection pool
  │   ├── schemas.cypher    # Contraintes + index (setup initial)
  │   └── entity_extractor.py  # Extraction entités depuis input/output
  ├── tools/
  │   ├── registry.py       # Tool registry + dispatch
  │   ├── search.py         # Web / RAG local (Ollama embeddings)
  │   ├── code_exec.py      # Sandbox Python (subprocess isolé)
  │   └── file_io.py        # Lecture/écriture fichiers
  ├── models/
  │   ├── ollama_client.py  # Appels directs Ollama /api/chat (pas de SDK tiers)
  │   └── router.py         # ModelRouter : E4B / 26B / 31B selon heuristique
  ├── infra/
  │   ├── state.py          # Cursor recovery + persistence atomique
  │   └── retry.py          # Retry exponentiel sur erreurs Ollama
  ├── agent.py              # Orchestrateur principal
  ├── config.yaml           # Tous les paramètres configurables
  └── AGENT_PATTERN.md      # Ce document
```

---

## 5. Configuration complète

```yaml
# config.yaml

models:
  triage:         gemma4:e4b       # Classification rapide, thinking off
  planner:        gemma4:26b       # MoE, thinking on
  executor:       gemma4:26b       # MoE, thinking adaptatif
  executor_draft: gemma4:e4b       # Speculative Draft mode
  critic:         gemma4:31b       # Dense, thinking on — évaluation rigoureuse
  critic_light:   gemma4:26b       # Fallback si RAM saturée
  ollama_base_url: http://localhost:11434

sampling:
  default:  { temperature: 1.0, top_p: 0.95, top_k: 64 }
  planner:  { temperature: 0.7, top_p: 0.95, top_k: 64 }
  critic:   { temperature: 0.3, top_p: 0.9,  top_k: 32 }

thinking:
  triage:   false
  planner:  true
  executor: false                  # true uniquement pour génération longue
  critic:   true

thresholds:
  min_score:        6.5            # Score Critic en-dessous = retry
  max_retries:      3              # Par step
  parallel_steps:   true           # asyncio.gather sur steps marqués parallel

skeleton:
  enabled:          true
  token_threshold:  500            # Activer SoT si expected_output > N tokens

speculative:
  enabled:          true
  max_input_tokens: 200            # Seuil d'activation du draft E4B
  tools_bypass:     ["code", "search"]  # Toujours 26B pour ces tools

vision:
  token_budget_ocr:    1120        # Documents, OCR, texte fin
  token_budget_default: 280        # Usage général
  token_budget_fast:    70         # Classification, captioning

recovery:
  persist_cursor:   true           # Sauvegarder l'avancement step par step
  state_file:       ./state/agent_state.json
  atomic_write:     true           # write .tmp + rename

memory:
  neo4j_uri:        bolt://localhost:7687
  neo4j_user:       neo4j
  context_k:        5              # Top-K épisodes injectés dans le Planner
  write_threshold:  5.0            # Ne persister que les épisodes score > N
  entity_extraction: true          # Extraire entités auto depuis input/output

context:
  max_tokens_per_call: 32768       # Budget par appel LLM (< 256K dispo)
  skeleton_threshold:  500         # Voir section skeleton
  trace_in_context:    true        # Inclure la trace des steps précédents

logging:
  level: INFO
  trace_file: ./logs/agent_trace.jsonl  # Log structuré de chaque épisode
```

---

## 6. Interfaces clés (contrats de code)

### agent.py — point d'entrée

```python
async def run(user_input: str) -> AgentResult:
    # 0. Triage — E4B, rapide, décide du pipeline
    complexity = await triage.classify(user_input)   # simple | complex | multimodal

    if complexity == "simple":
        return await executor.direct(user_input)     # E4B seul, pas de plan

    # 1. Memory read
    context = await memory.get_context(user_input)

    # 2. Plan (26B MoE, thinking on)
    plan = await planner.plan(user_input, context)
    state.save_plan(plan)

    # 3. Execute (séquentiel ou parallel selon plan.parallel)
    results = await executor.execute(plan, on_step_done=state.advance_cursor)

    # 4. Critique + retry (31B Dense pour high stakes, 26B sinon)
    scored = await critic.evaluate(plan, results)

    # 5. Memory write
    await memory.persist(user_input, plan, scored)

    return AgentResult(goal=plan.goal, results=scored)
```

### critic.py — boucle retry

```python
async def evaluate(plan: Plan, results: list[StepResult]) -> list[ScoredResult]:
    scored = []
    for step, result in zip(plan.steps, results):
        for attempt in range(config.max_retries):
            score = await _score(step, result)
            if score.final_score >= config.min_score:
                break
            result = await executor.execute_step(step)  # retry
        scored.append(ScoredResult(step=step, result=result, score=score))
    return scored
```

---

## 7. Analyse des compromis

| Dimension | API Cloud | Modèle unique local | Multi-modèle local (ce pattern) |
|---|---|---|---|
| Coût marginal | €€€ | €0 | €0 |
| Latence triage | 100ms | 3–15s (surdimensionné) | < 1s (E4B) |
| Latence raisonnement | 200–500ms | 3–15s | 3–15s (26B/31B) |
| Qualité brute | ★★★★★ | ★★★ | ★★★★ (scaffold + 31B Critic) |
| Souveraineté | Données ext. | 100% local | 100% local |
| Contexte long | 200K payant | 256K | 256K (26B/31B) |
| Mémoire persistante | Stateless | Neo4j | Neo4j |
| Multimodal | Text+Image | Text+Image | Text+Image+Audio (E4B) |
| Résilience | Dépend du réseau | Dépend du GPU | Retry + cursor recovery |

### Pourquoi pas de dépendance cloud ?

Ce pattern est conçu pour être **100% autonome**. Gemma 4 31B Dense (AIME 89.2%,
GPQA 84.3%) atteint un niveau de qualité suffisant pour le Critic sans API externe.
Le scaffold fort (plan structuré + retry + mémoire) compense l'écart résiduel avec
les modèles frontier cloud.

**Seul cas où reconsidérer :** raisonnement long-chain non décomposable sur des domaines
hors distribution du modèle. Dans ce cas, ajouter un fallback API est trivial
(même interface Ollama OpenAI-compatible).

---

## 8. Use cases prioritaires

### 8.1 Code review UrbaHive
- **Planner** : décompose le diff Git en chunks indépendants (par fichier / par fonction)
- **Executor** : analyse chaque chunk — sécurité, perf, patterns Neo4j, conventions
- **Critic** : note la qualité de la critique, retry si trop générique
- **Mémoire** : retient les patterns récurrents par module, les faux positifs historiques

Avantage clé : aucun code client n'est envoyé à une API externe.

### 8.2 Extraction docs EDF/Nuward (PTI/DEX)
- **Planner** : identifie les sections à extraire selon le schéma Polarion cible
- **Executor SoT** : traite chaque section en appel indépendant (256K contexte exploité)
- **Vision** : OCR des schémas techniques avec budget tokens 1120 (résolution max)
- **Critic** : valide la conformité structurelle au schéma Capella (31B Dense, thinking on)
- **Output** : JSON structuré prêt pour import

### 8.3 Running coach (WhatsApp/Telegram)
- **Mémoire** : historique séances Garmin (Fenix 6), objectifs, blessures passées
- **Planner** : analyse la semaine d'entraînement vs objectif
- **Executor** : génère le plan J+7 personnalisé
- **Critic** : vérifie la progressivité de charge (règle des 10%), cohérence avec blessures

---

## 9. Étapes suivantes

- [ ] Benchmark local : `gemma4:26b` et `gemma4:31b` sur M4 Pro — tokens/s, RAM réelle, temps de swap
- [ ] Valider le triage E4B : classifier 50 inputs variés, mesurer précision et latence
- [ ] Implémenter `ollama_client.py` : appels directs `/api/chat` avec tools + thinking toggle
- [ ] Implémenter `router.py` : ModelRouter avec heuristiques de sélection
- [ ] Implémenter `planner.py` avec validation pydantic stricte du plan JSON
- [ ] Implémenter `critic.py` avec boucle retry — comparer 26B vs 31B comme Critic
- [ ] Implémenter `state.py` : cursor recovery + persistence atomique
- [ ] Connecter Neo4j : tester enrichissement contexte sur un use case réel
- [ ] Benchmark Critic : calibrer le seuil 6.5 sur 20 tâches échantillon (26B et 31B)
- [ ] Tester vision : extraction document avec budget tokens variable (280 vs 1120)
- [ ] Choisir le use case pilote : **code review UrbaHive** (recommandé, périmètre borné)

---

*Document vivant — mettre à jour `config.yaml` et les seuils Critic au fur et à mesure
des calibrations sur cas réels.*
