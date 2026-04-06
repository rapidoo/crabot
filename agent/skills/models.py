"""Pydantic models for the skills subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SecurityFinding(BaseModel):
    """A single finding from a security review."""

    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str  # prompt_injection, code_execution, data_exfiltration, obfuscation
    description: str
    location: str = ""  # line number or section reference


class SecurityReport(BaseModel):
    """Result of a security review on code or prompt content."""

    verdict: Literal["pass", "warn", "fail"] = "pending"
    score: float = Field(default=0.0, ge=0, le=10)
    findings: list[SecurityFinding] = []
    reviewed_at: str = ""


class SkillMetadata(BaseModel):
    """Metadata for a downloaded/managed skill."""

    name: str
    description: str = ""
    source_url: str = ""
    version: str = "1.0.0"
    triggers: list[str] = []  # TRIGGER WHEN patterns
    anti_triggers: list[str] = []  # DO NOT TRIGGER WHEN patterns
    security_report: SecurityReport | None = None
    status: Literal["pending", "approved", "rejected", "evolved"] = "pending"
    fetched_at: str = ""
    evolved_from: str | None = None  # parent skill name if evolved
    use_count: int = 0
    avg_score: float = 0.0


class Skill(BaseModel):
    """A complete skill with metadata and content."""

    metadata: SkillMetadata
    content: str  # Full SKILL.md markdown body (after front matter)
    local_path: str = ""  # Path on disk
