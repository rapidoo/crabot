# Pattern Agentique : Plan → Execute → Critique
**Compensation d'un modèle local faible par un scaffold fort**

*Le Bris Consulting — Avril 2025 — v1.0 Draft*
*Contexte : Gemma 4 26B MoE · MacBook M4 Pro 48GB · Ollama*

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

- Modèle backbone : Gemma 4 26B MoE (actifs ≃ 4B) via Ollama
- Hardware : Apple Silicon M4 Pro, 48 GB unifié, MLX ou llama.cpp comme runtime
- Pas de dépendance obligatoire à un orchestrateur spécifique
- Contexte disponible : 256K tokens — exploité pour la mémoire et les traces d'exécution

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

### 3.4 Speculative Draft + Verifier

Exploite les deux tailles disponibles dans 48 GB RAM : le petit modèle génère vite, le grand
vérifie et corrige si nécessaire.

```
Gemma 4 E4B (actifs ~800M)  ──►  draft rapide (~2s)  ──►  Gemma 4 26B MoE
                                                            vérification (~8s)
                                                            correction si besoin

Gain : ~60% de réduction de latence sur les steps simples
Usage : boucles Executor où la vitesse prime sur la qualité initiale
```

**Heuristique d'activation :** utiliser E4B en draft si `step.tool == "none"` et
`len(step.input) < 200 tokens`. Sinon, appel direct 26B.

### 3.5 Memory-Augmented Context (Neo4j)

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
  │   ├── ollama_client.py  # Wrapper Ollama API (OpenAI-compat)
  │   └── router.py         # Sélection backbone vs draft selon heuristique
  ├── agent.py              # Orchestrateur principal
  ├── config.yaml           # Tous les paramètres configurables
  └── AGENT_PATTERN.md      # Ce document
```

---

## 5. Configuration complète

```yaml
# config.yaml

models:
  planner:        gemma4:27b
  executor:       gemma4:27b
  executor_draft: gemma4:e4b       # Speculative Draft mode
  critic:         gemma4:27b
  ollama_base_url: http://localhost:11434

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
    # 1. Memory read
    context = await memory.get_context(user_input)

    # 2. Plan
    plan = await planner.plan(user_input, context)

    # 3. Execute (séquentiel ou parallel selon plan.parallel)
    results = await executor.execute(plan)

    # 4. Critique + retry intégré
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

| Dimension | API Cloud seul | Local seul | Pattern hybride |
|---|---|---|---|
| Coût marginal | €€€ à l'échelle | €0 (matériel fixé) | €0 local + € rare cloud |
| Latence | 100–500ms | 3–15s | 3–15s local |
| Qualité brute | ★★★★★ | ★★★ | ★★★★ (scaffold) |
| Souveraineté | Données ext. | 100% local | 100% local |
| Contexte long | 200K payant | 256K gratuit | 256K gratuit |
| Mémoire persistante | Stateless | Neo4j natif | Neo4j natif |
| Multimodal | Oui (payant) | Oui (natif) | Oui |

### Quand basculer sur API cloud ?

- Raisonnement long-chain non décomposable (analyse stratégique ouverte)
- Latence critique (< 1s) pour UX interactive temps réel
- Volume très faible + qualité critique : coût API négligeable
- **Critic final à fort enjeu** : utiliser Claude Opus comme ultime Critic sur les décisions importantes

Pattern recommandé : Gemma 4 local pour 90% des appels (volume, contexte, mémoire),
API cloud uniquement pour le Critic final sur les outputs à fort enjeu.

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
- **Critic** : valide la conformité structurelle au schéma Capella
- **Output** : JSON structuré prêt pour import

### 8.3 Running coach (WhatsApp/Telegram)
- **Mémoire** : historique séances Garmin (Fenix 6), objectifs, blessures passées
- **Planner** : analyse la semaine d'entraînement vs objectif
- **Executor** : génère le plan J+7 personnalisé
- **Critic** : vérifie la progressivité de charge (règle des 10%), cohérence avec blessures

---

## 9. Étapes suivantes

- [ ] Valider `gemma4:27b` via Ollama sur M4 Pro — mesurer tokens/s réels et VRAM utilisée
- [ ] Implémenter `planner.py` avec validation pydantic stricte du plan JSON
- [ ] Implémenter `critic.py` avec boucle retry et logging structuré
- [ ] Connecter Neo4j : tester enrichissement contexte sur un use case réel
- [ ] Benchmark Critic : calibrer le seuil 6.5 sur 20 tâches échantillon
- [ ] Évaluer `gemma4:e4b` comme draft rapide — mesurer gain latence réel
- [ ] Choisir le use case pilote : **code review UrbaHive** (recommandé, périmètre borné)

---

*Document vivant — mettre à jour `config.yaml` et les seuils Critic au fur et à mesure
des calibrations sur cas réels.*
