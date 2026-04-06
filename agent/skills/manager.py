"""Skill manager — orchestrates discovery, security review, and evolution of skills."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent.skills.models import SecurityReport, Skill, SkillMetadata
from agent.skills.parser import extract_code_blocks, extract_triggers, parse_skill_md
from agent.skills.store import SkillStore

if TYPE_CHECKING:
    from agent.models.ollama_client import OllamaClient
    from agent.models.router import ModelRouter
    from agent.tools.security_review import SecurityReviewTool
    from agent.tools.spider_search import SpiderSearchTool

logger = logging.getLogger(__name__)


class SkillManager:
    """Central orchestrator for the skill lifecycle.

    Responsibilities:
    - Discover skills from URLs or web search (via spider_search)
    - Run security review before integration
    - Store approved skills locally
    - Match skills to user input via trigger patterns
    - Evolve skill prompts based on feedback
    """

    def __init__(
        self,
        store: SkillStore,
        security_tool: SecurityReviewTool,
        spider_tool: SpiderSearchTool | None = None,
        client: OllamaClient | None = None,
        router: ModelRouter | None = None,
        min_security_score: float = 7.0,
    ):
        self._store = store
        self._security = security_tool
        self._spider = spider_tool
        self._client = client
        self._router = router
        self._min_security_score = min_security_score

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover_from_url(self, url: str) -> list[Skill]:
        """Fetch skills from a GitHub repo or direct SKILL.md URL.

        Supports:
        - Direct SKILL.md URLs → single skill
        - GitHub repo URLs → tries to find marketplace.json, then individual skills
        """
        if self._spider is None:
            return []

        now = datetime.now(timezone.utc).isoformat()
        skills: list[Skill] = []

        # Convert GitHub repo URL to raw content URLs
        raw_url = self._github_to_raw(url)
        content = await self._spider.run(raw_url or url)

        if content.startswith("ERROR"):
            logger.warning("Failed to fetch %s: %s", url, content[:200])
            return []

        # Try parsing as marketplace.json
        if "marketplace.json" in url or "plugins" in content:
            skills.extend(await self._parse_marketplace(content, url, now))
        # Try parsing as SKILL.md directly
        elif "---" in content[:50]:
            skill = self._parse_single_skill(content, url, now)
            if skill:
                skills.append(skill)
        # Try to discover skill paths from page content
        else:
            skills.extend(await self._discover_from_page(content, url, now))

        logger.info("Discovered %d skills from %s", len(skills), url)
        return skills

    async def discover_from_search(self, query: str) -> list[Skill]:
        """Search the web for skills and attempt to fetch them."""
        if self._spider is None:
            return []

        search_query = f"SKILL.md agent skill {query}"
        results = await self._spider.run(search_query)

        if results.startswith("ERROR"):
            logger.warning("Search failed: %s", results[:200])
            return []

        # Extract URLs from search results
        urls = re.findall(r"https?://\S+", results)
        # Filter for likely skill sources
        skill_urls = [
            u for u in urls
            if any(kw in u.lower() for kw in ("skill", "github.com", "gitlab.com"))
        ]

        all_skills: list[Skill] = []
        for url in skill_urls[:5]:  # Limit to 5 URLs
            discovered = await self.discover_from_url(url)
            all_skills.extend(discovered)

        return all_skills

    # ------------------------------------------------------------------
    # Security Review & Integration
    # ------------------------------------------------------------------

    async def review_and_integrate(self, skill: Skill) -> Skill:
        """Run security review, approve/reject, and store if approved."""
        # Build review input
        review_input = f"type:skill|description:{skill.metadata.description}|content:{skill.content}"
        report_json = await self._security.run(review_input)

        try:
            report = SecurityReport.model_validate_json(report_json)
        except Exception:
            report = SecurityReport(
                verdict="warn", score=5.0,
                findings=[], reviewed_at=datetime.now(timezone.utc).isoformat(),
            )

        skill.metadata.security_report = report

        # Also review any embedded code blocks
        code_blocks = extract_code_blocks(skill.content)
        for lang, code in code_blocks:
            code_input = f"type:code|description:embedded {lang} code|content:{code}"
            code_report_json = await self._security.run(code_input)
            try:
                code_report = SecurityReport.model_validate_json(code_report_json)
                # Merge findings into main report
                report.findings.extend(code_report.findings)
                if code_report.verdict == "fail":
                    report.verdict = "fail"
                report.score = min(report.score, code_report.score)
            except Exception:
                pass

        # Decision
        if report.verdict == "fail" or report.score < self._min_security_score:
            skill.metadata.status = "rejected"
            logger.warning(
                "Skill %s REJECTED (verdict=%s, score=%.1f)",
                skill.metadata.name, report.verdict, report.score,
            )
        else:
            skill.metadata.status = "approved"
            await self._store.save(skill)
            logger.info(
                "Skill %s APPROVED (score=%.1f)",
                skill.metadata.name, report.score,
            )

        return skill

    async def bulk_review_and_integrate(self, skills: list[Skill]) -> dict[str, int]:
        """Review and integrate multiple skills. Returns stats."""
        stats = {"discovered": len(skills), "approved": 0, "rejected": 0}
        for skill in skills:
            result = await self.review_and_integrate(skill)
            if result.metadata.status == "approved":
                stats["approved"] += 1
            else:
                stats["rejected"] += 1
        return stats

    # ------------------------------------------------------------------
    # Matching & Retrieval
    # ------------------------------------------------------------------

    async def get_active_skills(self) -> list[Skill]:
        """Return all approved/evolved skills."""
        return await self._store.list_approved()

    async def match_skills(self, user_input: str) -> list[Skill]:
        """Find skills whose triggers match the user input."""
        active = await self.get_active_skills()
        input_lower = user_input.lower()
        matched: list[Skill] = []

        for skill in active:
            # Check anti-triggers first
            if any(anti in input_lower for anti in skill.metadata.anti_triggers):
                continue

            # Check triggers
            if skill.metadata.triggers:
                if any(trigger in input_lower for trigger in skill.metadata.triggers):
                    matched.append(skill)
            else:
                # No explicit triggers — match on skill name/description keywords
                name_words = skill.metadata.name.replace("-", " ").split()
                desc_words = skill.metadata.description.lower().split()[:10]
                keywords = set(name_words + desc_words)
                if any(kw in input_lower for kw in keywords if len(kw) > 3):
                    matched.append(skill)

        return matched

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    async def evolve_skill(self, skill_name: str, feedback: str) -> Skill | None:
        """Evolve a skill's content based on feedback.

        1. Load current skill
        2. Use LLM to suggest improvements
        3. Security review the evolution
        4. Version old, store new
        """
        skill = await self._store.load(skill_name)
        if skill is None:
            logger.warning("Cannot evolve: skill %s not found", skill_name)
            return None

        if not self._client or not self._router:
            logger.warning("Cannot evolve: no LLM client available")
            return None

        # Generate improved version
        model = self._router.select("planner")
        sampling = self._router.sampling("planner")

        prompt = (
            f"You are improving an AI skill definition. "
            f"The current skill is below, followed by feedback.\n\n"
            f"Current SKILL.md:\n---\n{skill.content}\n---\n\n"
            f"Feedback: {feedback}\n\n"
            f"Produce an improved version of the skill content. "
            f"Keep the same YAML front matter format. "
            f"Output ONLY the improved SKILL.md content, nothing else."
        )

        resp = await self._client.chat(
            model,
            [{"role": "user", "content": prompt}],
            sampling=sampling,
            thinking=True,
        )

        new_content = resp.content.strip()
        if not new_content:
            return None

        # Security review the evolution
        evolved_skill = Skill(
            metadata=SkillMetadata(
                name=skill.metadata.name,
                description=skill.metadata.description,
                source_url=skill.metadata.source_url,
                version=f"{skill.metadata.version}+evolved",
                triggers=skill.metadata.triggers,
                anti_triggers=skill.metadata.anti_triggers,
                status="pending",
                fetched_at=skill.metadata.fetched_at,
                evolved_from=skill.metadata.name,
            ),
            content=new_content,
        )

        reviewed = await self.review_and_integrate(evolved_skill)
        if reviewed.metadata.status == "approved":
            # Version the old skill before overwriting
            await self._store.version(skill_name)
            reviewed.metadata.status = "evolved"
            await self._store.save(reviewed)
            logger.info("Skill %s evolved successfully", skill_name)
            return reviewed

        logger.warning("Skill evolution rejected by security review")
        return None

    async def record_usage(self, skill_name: str, score: float) -> None:
        """Record a usage event for a skill (for evolution tracking)."""
        await self._store.update_stats(skill_name, score)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _github_to_raw(self, url: str) -> str | None:
        """Convert GitHub URLs to raw content URLs."""
        # https://github.com/owner/repo → raw.githubusercontent.com/owner/repo/main
        m = re.match(
            r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+)(/.*)?)?$",
            url,
        )
        if m:
            owner, repo, branch, path = m.groups()
            branch = branch or "main"
            path = path or ""
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}{path}"
        return None

    def _parse_single_skill(
        self, content: str, source_url: str, fetched_at: str
    ) -> Skill | None:
        """Parse a single SKILL.md content into a Skill object."""
        metadata_dict, body = parse_skill_md(content)
        if not metadata_dict.get("name"):
            # Try to infer name from URL
            name = re.sub(r"[^a-z0-9]+", "-", source_url.split("/")[-2].lower()) if "/" in source_url else "unknown"
            metadata_dict.setdefault("name", name)

        description = metadata_dict.get("description", "")
        triggers, anti_triggers = extract_triggers(description)

        metadata = SkillMetadata(
            name=metadata_dict["name"],
            description=description,
            source_url=source_url,
            triggers=triggers,
            anti_triggers=anti_triggers,
            fetched_at=fetched_at,
        )

        return Skill(metadata=metadata, content=body or content)

    async def _parse_marketplace(
        self, content: str, base_url: str, fetched_at: str
    ) -> list[Skill]:
        """Parse a marketplace.json and fetch individual skills."""
        skills: list[Skill] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return []
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return []

        # Extract skill paths from marketplace format
        for plugin in data.get("plugins", []):
            for skill_path in plugin.get("skills", []):
                # Resolve relative path to full URL
                skill_url = self._resolve_skill_url(base_url, skill_path)
                if skill_url and self._spider:
                    skill_content = await self._spider.run(skill_url)
                    if not skill_content.startswith("ERROR"):
                        skill = self._parse_single_skill(
                            skill_content, skill_url, fetched_at
                        )
                        if skill:
                            skills.append(skill)

        return skills

    async def _discover_from_page(
        self, content: str, base_url: str, fetched_at: str
    ) -> list[Skill]:
        """Try to discover SKILL.md links from a page's content."""
        skills: list[Skill] = []
        # Look for links to SKILL.md or skills/ directories
        skill_paths = re.findall(
            r"(?:skills/[a-z0-9_-]+/SKILL\.md|skills/[a-z0-9_-]+)", content
        )
        seen: set[str] = set()
        for path in skill_paths[:10]:
            if path in seen:
                continue
            seen.add(path)
            if not path.endswith("SKILL.md"):
                path = f"{path}/SKILL.md"
            skill_url = self._resolve_skill_url(base_url, path)
            if skill_url and self._spider:
                skill_content = await self._spider.run(skill_url)
                if not skill_content.startswith("ERROR"):
                    skill = self._parse_single_skill(
                        skill_content, skill_url, fetched_at
                    )
                    if skill:
                        skills.append(skill)

        return skills

    def _resolve_skill_url(self, base_url: str, relative_path: str) -> str | None:
        """Resolve a relative skill path against a base URL."""
        relative_path = relative_path.lstrip("./")
        # For GitHub, construct raw URL
        m = re.match(r"(https?://(?:raw\.)?githubusercontent\.com/[^/]+/[^/]+/[^/]+)/", base_url)
        if m:
            return f"{m.group(1)}/{relative_path}"

        # For GitHub repo URLs
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", base_url)
        if m:
            owner, repo = m.group(1), m.group(2)
            return f"https://raw.githubusercontent.com/{owner}/{repo}/main/{relative_path}"

        # Generic: join base + relative
        base = base_url.rsplit("/", 1)[0] if not base_url.endswith("/") else base_url.rstrip("/")
        return f"{base}/{relative_path}"
