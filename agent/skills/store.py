"""Skill store — local filesystem storage for downloaded skills."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent.skills.models import Skill, SkillMetadata

logger = logging.getLogger(__name__)


class SkillStore:
    """Manages skill persistence on disk.

    Layout::

        library_dir/
          <skill-name>/
            SKILL.md        # Original or evolved content
            metadata.json   # Serialized SkillMetadata
            versions/       # Backup versions before evolution
              v1_SKILL.md
              v1_metadata.json
    """

    def __init__(self, library_dir: str | Path = "agent/skills/library"):
        self._root = Path(library_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(self, skill: Skill) -> Path:
        """Save a skill to disk. Overwrites if exists."""
        skill_dir = self._root / skill.metadata.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill.content, encoding="utf-8")

        # Write metadata
        meta_path = skill_dir / "metadata.json"
        meta_path.write_text(
            skill.metadata.model_dump_json(indent=2), encoding="utf-8"
        )

        # Update local_path
        skill.local_path = str(skill_dir)
        logger.info("Skill saved: %s → %s", skill.metadata.name, skill_dir)
        return skill_dir

    async def load(self, name: str) -> Skill | None:
        """Load a skill from disk by name."""
        skill_dir = self._root / name
        skill_md_path = skill_dir / "SKILL.md"
        meta_path = skill_dir / "metadata.json"

        if not skill_md_path.exists() or not meta_path.exists():
            return None

        content = skill_md_path.read_text(encoding="utf-8")
        meta_raw = meta_path.read_text(encoding="utf-8")
        metadata = SkillMetadata.model_validate_json(meta_raw)

        return Skill(
            metadata=metadata,
            content=content,
            local_path=str(skill_dir),
        )

    async def list_all(self) -> list[Skill]:
        """List all stored skills."""
        skills: list[Skill] = []
        if not self._root.exists():
            return skills

        for skill_dir in sorted(self._root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill = await self.load(skill_dir.name)
            if skill is not None:
                skills.append(skill)
        return skills

    async def list_approved(self) -> list[Skill]:
        """List skills with status 'approved' or 'evolved'."""
        all_skills = await self.list_all()
        return [s for s in all_skills if s.metadata.status in ("approved", "evolved")]

    async def version(self, name: str) -> int:
        """Create a backup of the current version before evolution.

        Returns the backup version number.
        """
        skill_dir = self._root / name
        versions_dir = skill_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        # Determine version number
        existing = list(versions_dir.glob("v*_SKILL.md"))
        version = len(existing) + 1

        skill_md = skill_dir / "SKILL.md"
        meta_json = skill_dir / "metadata.json"

        if skill_md.exists():
            shutil.copy2(skill_md, versions_dir / f"v{version}_SKILL.md")
        if meta_json.exists():
            shutil.copy2(meta_json, versions_dir / f"v{version}_metadata.json")

        logger.info("Skill %s versioned: v%d", name, version)
        return version

    async def delete(self, name: str) -> bool:
        """Remove a skill from the store."""
        skill_dir = self._root / name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info("Skill deleted: %s", name)
            return True
        return False

    async def update_stats(
        self, name: str, score: float
    ) -> None:
        """Update usage count and running average score for a skill."""
        skill = await self.load(name)
        if skill is None:
            return

        meta = skill.metadata
        total = meta.avg_score * meta.use_count + score
        meta.use_count += 1
        meta.avg_score = round(total / meta.use_count, 2)

        meta_path = self._root / name / "metadata.json"
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
