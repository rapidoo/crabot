"""SKILL.md parser — extracts YAML front matter and markdown body."""

from __future__ import annotations

import logging
import re

import yaml

logger = logging.getLogger(__name__)


def parse_skill_md(raw: str) -> tuple[dict, str]:
    """Parse a SKILL.md file into (metadata_dict, body_markdown).

    Expected format::

        ---
        name: my-skill
        description: "What this skill does"
        ---
        # Skill instructions in markdown ...

    Returns ({name, description, ...}, body) or ({}, raw) on parse failure.
    """
    raw = raw.strip()
    if not raw.startswith("---"):
        return {}, raw

    # Find the closing --- delimiter
    end = raw.find("---", 3)
    if end == -1:
        return {}, raw

    front_matter = raw[3:end].strip()
    body = raw[end + 3:].strip()

    try:
        metadata = yaml.safe_load(front_matter)
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse SKILL.md front matter: %s", exc)
        metadata = {}

    return metadata, body


def extract_triggers(description: str) -> tuple[list[str], list[str]]:
    """Extract trigger and anti-trigger patterns from a skill description.

    Looks for patterns like:
        TRIGGER when: condition1, condition2
        DO NOT TRIGGER when: condition1, condition2

    Returns (triggers, anti_triggers) as lists of lowercase strings.
    """
    triggers: list[str] = []
    anti_triggers: list[str] = []

    if not description:
        return triggers, anti_triggers

    # Extract DO NOT TRIGGER first (must come before TRIGGER to avoid partial match)
    anti_pattern = re.compile(
        r"(?:DO\s+NOT\s+TRIGGER|don'?t\s+trigger)\s+(?:when|if)\s*[:\-]?\s*(.+?)(?:\.|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in anti_pattern.finditer(description):
        raw_triggers = match.group(1).strip()
        anti_triggers.extend(_split_trigger_list(raw_triggers))

    # Extract TRIGGER WHEN
    trigger_pattern = re.compile(
        r"(?<!NOT\s)TRIGGER\s+(?:when|if)\s*[:\-]?\s*(.+?)(?:\.|DO\s+NOT|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in trigger_pattern.finditer(description):
        raw_triggers = match.group(1).strip()
        triggers.extend(_split_trigger_list(raw_triggers))

    return triggers, anti_triggers


def _split_trigger_list(raw: str) -> list[str]:
    """Split a trigger string on commas or 'or' into individual conditions."""
    # Split on comma or " or "
    parts = re.split(r",\s*|\s+or\s+", raw, flags=re.IGNORECASE)
    return [p.strip().lower() for p in parts if p.strip()]


def extract_code_blocks(content: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks from markdown content.

    Returns list of (language, code) tuples.
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    return [(m.group(1) or "text", m.group(2).strip()) for m in pattern.finditer(content)]
