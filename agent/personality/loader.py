"""Personality loader — loads workspace Markdown files for prompt injection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path(__file__).parent.parent.parent / "workspace"


@dataclass
class Identity:
    name: str = "Agent"
    emoji: str = "🤖"
    creature: str = ""
    vibe: str = ""
    theme: str = ""


@dataclass
class Personality:
    soul: str = ""
    identity: Identity = field(default_factory=Identity)
    agents: str = ""
    user: str = ""
    heartbeat: str = ""


def load_personality(workspace_dir: Path | None = None) -> Personality:
    """Load all workspace Markdown files into a Personality object."""
    ws = workspace_dir or _DEFAULT_WORKSPACE
    p = Personality()

    p.soul = _read_file(ws / "SOUL.md")
    p.agents = _read_file(ws / "AGENTS.md")
    p.user = _read_file(ws / "USER.md")
    p.heartbeat = _read_file(ws / "HEARTBEAT.md")
    p.identity = _parse_identity(ws / "IDENTITY.md")

    loaded = [f for f in ("SOUL", "IDENTITY", "AGENTS", "USER", "HEARTBEAT")
              if (ws / f"{f}.md").exists()]
    if loaded:
        logger.info("Personality loaded: %s from %s", ", ".join(loaded), ws)

    return p


def _read_file(path: Path) -> str:
    """Read a file, return empty string if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _parse_identity(path: Path) -> Identity:
    """Parse IDENTITY.md into an Identity dataclass."""
    text = _read_file(path)
    if not text:
        return Identity()

    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("- ")
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()

    return Identity(
        name=fields.get("name", "Agent"),
        emoji=fields.get("emoji", "🤖"),
        creature=fields.get("creature", ""),
        vibe=fields.get("vibe", ""),
        theme=fields.get("theme", ""),
    )
