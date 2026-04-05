"""Meta-tool: tool_create — generates, validates, and registers new tools at runtime.

The agent can use this tool to create new tools that persist across sessions.
Generated tools are saved in agent/tools/custom/ and auto-discovered on next startup.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
import textwrap
from pathlib import Path

from agent.tools.base import Tool
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

CUSTOM_TOOLS_DIR = Path(__file__).parent / "custom"

_FORBIDDEN_MODULES = {"os", "subprocess", "shutil", "socket", "sys", "pathlib", "ctypes"}
_FORBIDDEN_BUILTINS = {"eval", "exec", "__import__", "compile", "globals", "locals"}


def _validate_ast(source: str) -> str | None:
    """Return error message if source contains forbidden constructs, else None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                    return f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _FORBIDDEN_MODULES:
                return f"Forbidden import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
                return f"Forbidden builtin: {node.func.id}"
    return None

_TOOL_TEMPLATE = '''\
"""Auto-generated tool: {name} — {description}"""

from __future__ import annotations

from agent.tools.registry import register_tool


class {class_name}:
    """Tool: {description}"""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def description(self) -> str:
        return "{description}"

    async def run(self, input: str) -> str:
{run_body}


def _factory() -> {class_name}:
    return {class_name}()


register_tool("{name}", _factory)
'''


def _sanitize_name(name: str) -> str:
    """Ensure a valid Python identifier for the tool name."""
    name = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "custom_tool"


def _to_class_name(name: str) -> str:
    """Convert tool_name to ToolNameTool."""
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts) + "Tool"


class ToolCreateTool:
    """Create new tools that persist across sessions.

    Input format (one of):
        name:<tool_name>|description:<desc>|code:<python_code_for_run_method>
        name:<tool_name>|description:<desc>|shell:<shell_command>
    """

    @property
    def name(self) -> str:
        return "tool_create"

    @property
    def description(self) -> str:
        return (
            "Create a new persistent tool. "
            "Input: name:<name>|description:<desc>|code:<async run body>"
        )

    async def run(self, input: str) -> str:
        """Parse the tool spec, generate the module, validate, and register."""
        try:
            spec = self._parse_spec(input)
        except ValueError as exc:
            return f"ERROR: {exc}"

        tool_name = _sanitize_name(spec["name"])
        class_name = _to_class_name(tool_name)
        description = spec.get("description", f"Custom tool: {tool_name}")

        # Build the run method body
        if "code" in spec:
            run_body = self._indent_code(spec["code"])
        elif "shell" in spec:
            run_body = self._shell_wrapper(spec["shell"])
        else:
            return "ERROR: must provide code:<...> or shell:<...>"

        # Generate the module source
        source = _TOOL_TEMPLATE.format(
            name=tool_name,
            class_name=class_name,
            description=description.replace('"', '\\"'),
            run_body=run_body,
        )

        # Validate by compiling
        try:
            compile(source, f"<tool:{tool_name}>", "exec")
        except SyntaxError as exc:
            return f"ERROR: generated code has syntax error: {exc}"

        # Validate AST for forbidden constructs
        ast_error = _validate_ast(source)
        if ast_error:
            return f"ERROR: {ast_error}"

        # Write to custom tools directory
        CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        module_path = CUSTOM_TOOLS_DIR / f"{tool_name}.py"
        module_path.write_text(source, encoding="utf-8")

        # Load and register immediately
        try:
            module_name = f"agent.tools.custom.{tool_name}"
            if module_name in importlib.import_module("sys").modules:
                importlib.reload(importlib.import_module(module_name))
            else:
                importlib.import_module(module_name)
        except Exception as exc:
            # Remove broken module
            module_path.unlink(missing_ok=True)
            return f"ERROR: tool loaded but failed to register: {exc}"

        logger.info("Tool created: %s → %s", tool_name, module_path)
        return (
            f"OK: tool '{tool_name}' created and registered.\n"
            f"File: {module_path}\n"
            f"Available immediately and persists across restarts."
        )

    def _parse_spec(self, input: str) -> dict[str, str]:
        """Parse name:...|description:...|code:... format."""
        spec: dict[str, str] = {}
        # Split on | but not inside code blocks
        parts = input.split("|")
        current_key = None
        current_val: list[str] = []

        for part in parts:
            matched = False
            for prefix in ("name:", "description:", "code:", "shell:"):
                if part.strip().startswith(prefix):
                    if current_key:
                        spec[current_key] = "|".join(current_val)
                    current_key = prefix[:-1]
                    current_val = [part.strip()[len(prefix):]]
                    matched = True
                    break
            if not matched and current_key:
                current_val.append(part)

        if current_key:
            spec[current_key] = "|".join(current_val)

        if "name" not in spec:
            raise ValueError("Missing name: field")

        return spec

    def _indent_code(self, code: str) -> str:
        """Indent user code to fit inside the run method."""
        lines = code.strip().splitlines()
        if not lines:
            return '        return "No implementation"'
        return "\n".join(f"        {line}" for line in lines)

    def _shell_wrapper(self, command: str) -> str:
        """Generate a run body that executes a shell command."""
        return textwrap.dedent(f'''\
        import asyncio, shlex
        args = shlex.split({command.strip()!r}) + [input]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        result = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            result += "\\nSTDERR: " + stderr.decode("utf-8", errors="replace").strip()
        return result''').replace("\n", "\n        ")


def _factory() -> ToolCreateTool:
    return ToolCreateTool()


register_tool("tool_create", _factory)
