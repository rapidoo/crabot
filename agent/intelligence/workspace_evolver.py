"""Workspace evolver — synthesizes lessons into workspace personality files.

Periodically reads accumulated lessons from Neo4j and proposes updates
to USER.md, AGENTS.md, and SOUL.md so the agent's personality and rules
evolve based on user corrections and performance insights.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agent.config import get_settings
from agent.models.ollama_client import OllamaClient
from agent.models.router import ModelRouter

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path(__file__).parent.parent.parent / "workspace"

# Which lesson categories map to which workspace files
_CATEGORY_FILE_MAP = {
    "preference": "USER.md",
    "correction": "AGENTS.md",
    "process": "AGENTS.md",
    "fact": "AGENTS.md",
}

_EVOLVE_SYSTEM_PROMPT = """\
You are a configuration file editor for an AI agent.
Your job is to integrate recent lessons (user corrections) into the agent's \
configuration file.

Rules:
- Keep the existing Markdown format (headings, bullet lists)
- ONLY add information that comes from the lessons — do not invent
- Do not remove existing content unless a lesson explicitly contradicts it
- Integrate naturally: add new bullet points in the right section
- Keep it concise — one line per lesson, same style as existing
- If no lesson is relevant for this file, return exactly: NO_CHANGE

Return ONLY the updated file content. No explanation, no code fences."""

_NO_CHANGE = "NO_CHANGE"


async def evolve_workspace(
    client: OllamaClient,
    router: ModelRouter,
    memory: object,
    workspace_dir: Path | None = None,
) -> list[str]:
    """Synthesize high-confidence lessons into workspace files.

    Returns list of modified file names (e.g. ["USER.md", "AGENTS.md"]).
    """
    ws = workspace_dir or _DEFAULT_WORKSPACE
    if not getattr(memory, "available", False):
        return []

    # 1. Fetch lessons with high confidence
    try:
        lessons = await memory.get_high_confidence_lessons(min_confidence=2.0, limit=20)
    except Exception as exc:
        logger.warning("Workspace evolution: failed to fetch lessons: %s", exc)
        return []

    if not lessons:
        logger.info("Workspace evolution: no high-confidence lessons, skipping")
        return []

    # 2. Group lessons by target file
    file_lessons: dict[str, list[dict]] = {}
    for lesson in lessons:
        category = lesson.get("category", "correction")
        target_file = _CATEGORY_FILE_MAP.get(category, "AGENTS.md")
        file_lessons.setdefault(target_file, []).append(lesson)

    # 3. For each file with lessons, ask LLM to produce updated version
    model = router.select("planner")
    sampling = router.sampling("planner")
    modified: list[str] = []

    for filename, file_lessons_list in file_lessons.items():
        filepath = ws / filename
        if not filepath.exists():
            continue

        current_content = filepath.read_text(encoding="utf-8").strip()
        lessons_text = _format_lessons(file_lessons_list)

        prompt = (
            f"Current file ({filename}):\n---\n{current_content}\n---\n\n"
            f"Lessons to integrate:\n{lessons_text}\n\n"
            f"Return the updated file content."
        )

        try:
            resp = await client.chat(
                model,
                [
                    {"role": "system", "content": _EVOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                sampling=sampling,
                thinking=False,
            )
            new_content = resp.content.strip()
        except Exception as exc:
            logger.warning("Workspace evolution: LLM call failed for %s: %s", filename, exc)
            continue

        # Skip if no change
        if new_content == _NO_CHANGE or not new_content:
            logger.info("Workspace evolution: no changes for %s", filename)
            continue

        # Strip code fences if LLM wrapped the output
        new_content = _strip_code_fences(new_content)

        # Sanity check: content shouldn't shrink drastically
        if len(new_content) < len(current_content) * 0.5:
            logger.warning(
                "Workspace evolution: %s shrank too much (%d → %d), skipping",
                filename, len(current_content), len(new_content),
            )
            continue

        # Skip if content is identical
        if new_content.strip() == current_content.strip():
            continue

        # Backup + write
        backup_path = filepath.with_suffix(".md.bak")
        shutil.copy2(filepath, backup_path)
        filepath.write_text(new_content + "\n", encoding="utf-8")
        modified.append(filename)
        logger.info("Workspace evolution: updated %s (%d lessons integrated)", filename, len(file_lessons_list))

    return modified


def _format_lessons(lessons: list[dict]) -> str:
    """Format lessons for the LLM prompt."""
    lines = []
    for lesson in lessons:
        rule = lesson.get("rule", "")
        context = lesson.get("context", "")
        category = lesson.get("category", "")
        confidence = lesson.get("confidence", 1.0)
        reinforced = lesson.get("times_reinforced", 0)
        line = f"- [{category}] {rule}"
        if context:
            line += f" (context: {context})"
        if reinforced > 0:
            line += f" (confirmed {reinforced + 1}x)"
        lines.append(line)
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
