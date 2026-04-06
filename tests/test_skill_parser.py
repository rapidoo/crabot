"""Tests for agent.skills.parser — SKILL.md parsing and trigger extraction."""

import pytest

from agent.skills.parser import extract_code_blocks, extract_triggers, parse_skill_md


class TestParseSkillMd:
    def test_valid_front_matter(self):
        raw = """---
name: my-skill
description: "A test skill"
---
# Instructions
Do something useful."""
        metadata, body = parse_skill_md(raw)
        assert metadata["name"] == "my-skill"
        assert metadata["description"] == "A test skill"
        assert "# Instructions" in body

    def test_no_front_matter(self):
        raw = "# Just markdown\nNo front matter here."
        metadata, body = parse_skill_md(raw)
        assert metadata == {}
        assert body == raw

    def test_empty_front_matter(self):
        raw = "---\n---\nBody content."
        metadata, body = parse_skill_md(raw)
        assert metadata == {}
        assert body == "Body content."

    def test_missing_closing_delimiter(self):
        raw = "---\nname: broken\nNo closing delimiter."
        metadata, body = parse_skill_md(raw)
        assert metadata == {}
        assert body == raw

    def test_extra_yaml_fields(self):
        raw = """---
name: pdf
description: "Handle PDFs"
license: MIT
version: "2.0"
---
Body here."""
        metadata, body = parse_skill_md(raw)
        assert metadata["name"] == "pdf"
        assert metadata["license"] == "MIT"
        assert metadata["version"] == "2.0"

    def test_whitespace_handling(self):
        raw = """
---
name: spaces
description: "test"
---

Body with leading newline."""
        metadata, body = parse_skill_md(raw)
        assert metadata["name"] == "spaces"
        assert "Body with leading newline" in body


class TestExtractTriggers:
    def test_basic_trigger(self):
        desc = "TRIGGER when: user asks about PDFs."
        triggers, anti = extract_triggers(desc)
        assert len(triggers) >= 1
        assert "user asks about pdfs" in triggers
        assert anti == []

    def test_trigger_with_anti(self):
        desc = (
            "TRIGGER when: code imports anthropic. "
            "DO NOT TRIGGER when: code imports openai."
        )
        triggers, anti = extract_triggers(desc)
        assert any("anthropic" in t for t in triggers)
        assert any("openai" in t for t in anti)

    def test_multiple_triggers(self):
        desc = "TRIGGER when: user wants PDF, Excel, or Word documents."
        triggers, _ = extract_triggers(desc)
        assert len(triggers) >= 2

    def test_empty_description(self):
        triggers, anti = extract_triggers("")
        assert triggers == []
        assert anti == []

    def test_no_trigger_keywords(self):
        desc = "This skill handles document processing."
        triggers, anti = extract_triggers(desc)
        assert triggers == []
        assert anti == []


class TestExtractCodeBlocks:
    def test_single_python_block(self):
        content = """Some text
```python
print("hello")
```
More text."""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0] == ("python", 'print("hello")')

    def test_multiple_blocks(self):
        content = """```javascript
console.log("hi")
```
Some text
```python
x = 1
```"""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 2
        assert blocks[0][0] == "javascript"
        assert blocks[1][0] == "python"

    def test_no_language(self):
        content = """```
plain code
```"""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0][0] == "text"

    def test_no_code_blocks(self):
        content = "Just regular markdown text."
        blocks = extract_code_blocks(content)
        assert blocks == []
