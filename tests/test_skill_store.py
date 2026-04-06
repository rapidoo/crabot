"""Tests for agent.skills.store — local filesystem skill storage."""

import shutil
from pathlib import Path

import pytest

from agent.skills.models import Skill, SkillMetadata
from agent.skills.store import SkillStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a SkillStore with a temporary directory."""
    store = SkillStore(library_dir=tmp_path / "skills")
    return store


def _make_skill(name: str = "test-skill", status: str = "approved") -> Skill:
    return Skill(
        metadata=SkillMetadata(
            name=name,
            description=f"Test skill: {name}",
            source_url="https://example.com/skills",
            status=status,
            fetched_at="2025-01-01T00:00:00Z",
        ),
        content=f"# {name}\nThis is a test skill.",
    )


class TestSkillStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_store):
        skill = _make_skill()
        path = await tmp_store.save(skill)

        assert path.exists()
        assert (path / "SKILL.md").exists()
        assert (path / "metadata.json").exists()

        loaded = await tmp_store.load("test-skill")
        assert loaded is not None
        assert loaded.metadata.name == "test-skill"
        assert "# test-skill" in loaded.content

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_store):
        result = await tmp_store.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, tmp_store):
        await tmp_store.save(_make_skill("skill-a"))
        await tmp_store.save(_make_skill("skill-b"))
        await tmp_store.save(_make_skill("skill-c", status="rejected"))

        all_skills = await tmp_store.list_all()
        assert len(all_skills) == 3

    @pytest.mark.asyncio
    async def test_list_approved(self, tmp_store):
        await tmp_store.save(_make_skill("good", status="approved"))
        await tmp_store.save(_make_skill("evolved", status="evolved"))
        await tmp_store.save(_make_skill("bad", status="rejected"))

        approved = await tmp_store.list_approved()
        assert len(approved) == 2
        names = {s.metadata.name for s in approved}
        assert names == {"good", "evolved"}

    @pytest.mark.asyncio
    async def test_version(self, tmp_store):
        await tmp_store.save(_make_skill("versioned"))
        v = await tmp_store.version("versioned")
        assert v == 1

        # Version again
        v2 = await tmp_store.version("versioned")
        assert v2 == 2

        # Check backup files exist
        versions_dir = tmp_store._root / "versioned" / "versions"
        assert (versions_dir / "v1_SKILL.md").exists()
        assert (versions_dir / "v2_SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_delete(self, tmp_store):
        await tmp_store.save(_make_skill("to-delete"))
        assert await tmp_store.load("to-delete") is not None

        deleted = await tmp_store.delete("to-delete")
        assert deleted is True
        assert await tmp_store.load("to-delete") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_store):
        deleted = await tmp_store.delete("nope")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_stats(self, tmp_store):
        await tmp_store.save(_make_skill("tracked"))

        await tmp_store.update_stats("tracked", 8.0)
        skill = await tmp_store.load("tracked")
        assert skill.metadata.use_count == 1
        assert skill.metadata.avg_score == 8.0

        await tmp_store.update_stats("tracked", 6.0)
        skill = await tmp_store.load("tracked")
        assert skill.metadata.use_count == 2
        assert skill.metadata.avg_score == 7.0
