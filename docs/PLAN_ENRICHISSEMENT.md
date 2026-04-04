# Plan : Enrichissement AGENT_PATTERN.md avec patterns NanoClaw + specs Gemma 4

## Contexte

L'utilisateur a un document `AGENT_PATTERN.md` décrivant un pattern agentique Plan→Execute→Critique
pour Gemma 4 sur Ollama (MacBook M4 Pro 48GB). Il veut l'enrichir avec :

1. Les patterns architecturaux appris de NanoClaw (isolation, IPC, concurrence, registry)
2. Les specs réelles de Gemma 4 (corrigées depuis la page Ollama officielle)
3. La possibilité de mixer plusieurs modèles selon les besoins

**Approche** : sobre et élégante — pas de réécriture totale, des enrichissements ciblés qui
renforcent le document existant.

## Livrable

**Fichier modifié** : `AGENT_PATTERN.md` (enrichi, pas remplacé)

## Enrichissements prévus

### A. Corrections factuelles Gemma 4 (section 1 + config)

Le doc mentionne "Gemma 4 26B MoE" — il existe en réalité 4 variantes avec des profils distincts :

| Variante | Params | Actifs | Contexte | Modalities | RAM Ollama | Rôle agent |
|----------|--------|--------|----------|------------|------------|------------|
| E2B | 5.1B (2.3B eff) | 2.3B | 128K | Text, Image, Audio | 7.2 GB | Draft ultra-rapide, triage |
| E4B | 8B (4.5B eff) | 4.5B | 128K | Text, Image, Audio | 9.6 GB | Draft rapide, classification |
| 26B MoE | 25.2B | 3.8B | 256K | Text, Image | 18 GB | Backbone principal (doc actuel) |
| 31B Dense | 30.7B | 30.7B | 256K | Text, Image | 20 GB | Critic fort, raisonnement complexe |

**Impact** : le doc utilise E4B comme draft et 26B comme backbone — correct.
Mais le 31B Dense est un bien meilleur Critic (AIME 89.2% vs 88.3%, GPQA 84.3% vs 82.3%).
Sur 48GB M4 Pro : 26B (18GB) + E4B (9.6GB) = 27.6 GB — reste 20GB pour le 31B Dense.
**Question RAM** : les 3 ne tiennent pas simultanément. Router entre 2 modèles max en mémoire.

### B. Thinking Mode natif (nouvelle section 3.x)

Gemma 4 a un mode thinking natif activé via `<|think|>` en début de system prompt.
Le modèle génère son raisonnement interne dans des blocs `<|channel>thought\n...<channel|>`.

**Pattern** : activer le thinking pour le Planner et le Critic (raisonnement structuré),
le désactiver pour l'Executor (exécution directe, vitesse).

### C. Stratégie multi-modèle enrichie (nouvelle section ou enrichissement 3.4)

Inspiré du channel registry de NanoClaw : un routeur de modèles avec factory pattern.

```
Triage (E4B, thinking off)
  → Classification rapide de la tâche
  → Sélection du pipeline : simple | complexe | multimodal

Simple : E4B seul (thinking off) — réponse directe
Complexe : 26B MoE (thinking on) → Plan → Execute → 31B Critic (thinking on)
Multimodal : 26B/31B pour vision, E4B pour audio
```

Heuristique de routage :
- Longueur input < 100 tokens + pas d'image → E4B direct
- Tâche structurable (code, extraction) → 26B MoE Plan+Execute
- Évaluation critique / décision importante → 31B Dense Critic
- Image/document → 26B ou 31B (vision encoder 550M)
- Audio → E4B ou E2B uniquement (seuls à avoir l'audio encoder)

### D. Patterns NanoClaw transposés (nouvelle section 3.6 ou annexe)

#### D.1 Tool Registry (inspiré de channels/registry.ts)
```python
# Pattern factory avec null = skip (outil non disponible)
tool_registry: dict[str, ToolFactory] = {}

def register_tool(name: str, factory: Callable) -> None:
    tool_registry[name] = factory

def get_tool(name: str) -> Tool | None:
    factory = tool_registry.get(name)
    return factory() if factory else None  # None = pas dispo, skip
```
Avantage vs le registry actuel du doc : découplage total, ajout de tools sans toucher l'orchestrateur.

#### D.2 IPC file-based (inspiré du protocole IPC NanoClaw)
Pour les exécutions longues (code sandbox, RAG), communication par fichiers atomiques :
```
ipc/
  input/    ← orchestrateur écrit (atomic rename)
  output/   ← worker écrit (atomic rename)
  _close    ← sentinelle d'arrêt
```
Plus robuste que stdin/stdout pour les cas multi-process. Pattern : écrire `.tmp` puis `rename()`.

#### D.3 Concurrence bornée (inspiré de GroupQueue)
Si l'agent gère plusieurs tâches simultanées (ex: code review multi-fichier) :
- Bounded semaphore sur les appels Ollama (1 seul modèle actif en inférence GPU)
- File d'attente avec priorité (tâches Critic > tâches Executor > tâches Draft)
- Retry exponentiel sur les échecs Ollama (OOM, timeout)

#### D.4 Cursor recovery (inspiré du message tracking NanoClaw)
Persister l'état d'avancement du plan dans SQLite/Neo4j :
- Chaque step complété = cursor avancé + persisté
- Crash recovery : reprendre au dernier step réussi, pas depuis zéro
- Stale session detection : si le state file est corrompu, reset propre

### E. Sampling parameters (best practices Gemma 4)

Le doc ne mentionne pas les params de sampling recommandés par Google :
- `temperature=1.0`, `top_p=0.95`, `top_k=64`
- Pour le Critic : possiblement `temperature=0.3` pour plus de déterminisme

### F. Variable image resolution (enrichissement multimodal)

Gemma 4 supporte des budgets de tokens visuels : 70, 140, 280, 560, 1120.
- OCR/documents → budget élevé (1120)
- Classification/captioning → budget bas (70-140)
Configurable par step dans le plan.

## Structure des modifications dans le fichier

1. **Section 1** : Mettre à jour le tableau des variantes Gemma 4, mentionner le 31B Dense
2. **Section 3.4** : Enrichir "Speculative Draft" avec la stratégie multi-modèle complète (E4B/26B/31B)
3. **Nouvelle section 3.5bis** : "Thinking Mode natif" — quand activer/désactiver
4. **Section 3.6 (nouvelle)** : "Patterns d'infrastructure" — Tool Registry, IPC, Concurrence, Recovery (inspirés NanoClaw)
5. **Section 5** : Enrichir config.yaml avec sampling params, thinking toggle, variable vision budget, model router
6. **Section 7** : Tableau de compromis — ajouter colonne 31B Dense vs 26B MoE

## Étapes d'implémentation

1. Pull le repo distant pour avoir AGENT_PATTERN.md localement
2. Appliquer les enrichissements ci-dessus par éditions ciblées
3. Commit et push sur `claude/nanoclaw-architecture-docs-e9TKh`

## Conventions

- Garder le ton et le style du document existant (technique, concis, pas de prose superflue)
- Français pour la prose, anglais pour le code et les termes techniques
- Pas de sections redondantes — enrichir l'existant plutôt qu'ajouter des pavés
- Sobre et élégant : chaque ajout doit apporter de la valeur actionnable

## Vérification

- Le document reste cohérent de bout en bout
- Les specs Gemma 4 sont correctes (vérifiées vs page Ollama)
- Les patterns NanoClaw sont correctement transposés (pas de copier-coller TypeScript→Python)
- La config.yaml reste fonctionnelle et complète
- Le fichier Markdown rend correctement sur GitHub
