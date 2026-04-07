"""Evolution worker — applies code mutations and validates before reload.

Short-lived subprocess spawned by the supervisor:
    python -m agent.worker_evolution --mutations '[{...}]'

Applies mutations to source files, validates each one, and reports
success/failure as JSON on stdout.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate_syntax(file_path: Path) -> str | None:
    """Check if a Python file has valid syntax. Returns error or None."""
    try:
        source = file_path.read_text()
        compile(source, str(file_path), "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError in {file_path}: {exc}"


def _validate_import(file_path: Path) -> str | None:
    """Check if a module can be imported. Returns error or None."""
    # Convert file path to module name
    rel = file_path.relative_to(REPO_ROOT)
    parts = list(rel.with_suffix("").parts)
    module_name = ".".join(parts)
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        importlib.import_module(module_name)
        return None
    except Exception as exc:
        return f"ImportError for {module_name}: {exc}"


def _smoke_test() -> str | None:
    """Quick smoke test: can we instantiate the Agent and run triage?"""
    try:
        # Fresh import to pick up changes
        for mod_name in list(sys.modules):
            if mod_name.startswith("agent."):
                del sys.modules[mod_name]
        from agent.agent import Agent
        from agent.config import get_settings
        # Just instantiate — don't connect to Neo4j
        Agent(get_settings())
        return None
    except Exception as exc:
        return f"Smoke test failed: {exc}"


def apply_and_validate(mutations: list[dict]) -> dict:
    """Apply mutations to source files and validate.

    Each mutation: {"target": "path/to/file.py", "new_value": "content"}

    Returns {"ok": True/False, "error": "...", "applied": [...], "rolled_back": [...]}
    """
    applied: list[str] = []
    backups: dict[str, Path] = {}      # target -> backup path (existed before)
    created: list[str] = []            # targets that didn't exist before

    for mut in mutations:
        target = mut.get("target", "")
        new_value = mut.get("new_value", "")

        target_path = (REPO_ROOT / target).resolve()

        # Safety: must stay in repo
        if not str(target_path).startswith(str(REPO_ROOT)):
            return {"ok": False, "error": f"Path escapes repo: {target}", "applied": applied, "rolled_back": []}

        # Backup original or track as new
        if target_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + ".evo_bak")
            shutil.copy2(target_path, backup_path)
            backups[target] = backup_path
        else:
            created.append(target)

        # Write new content
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_value)
        applied.append(target)

        # Validate syntax
        if target_path.suffix == ".py":
            err = _validate_syntax(target_path)
            if err:
                _rollback(backups, created)
                return {"ok": False, "error": err, "applied": [], "rolled_back": list(backups.keys()) + created}

    # Validate imports for all modified Python files
    for target in applied:
        target_path = (REPO_ROOT / target).resolve()
        if target_path.suffix == ".py":
            err = _validate_import(target_path)
            if err:
                _rollback(backups, created)
                return {"ok": False, "error": err, "applied": [], "rolled_back": list(backups.keys()) + created}

    # Smoke test — can the agent still load?
    err = _smoke_test()
    if err:
        _rollback(backups, created)
        return {"ok": False, "error": err, "applied": [], "rolled_back": list(backups.keys()) + created}

    # Clean up backups on success
    for backup_path in backups.values():
        backup_path.unlink(missing_ok=True)

    return {"ok": True, "applied": applied, "rolled_back": []}


def _rollback(backups: dict[str, Path], created: list[str] | None = None) -> None:
    """Restore original files from backups and delete newly created files."""
    for target, backup_path in backups.items():
        target_path = (REPO_ROOT / target).resolve()
        if backup_path.exists():
            shutil.copy2(backup_path, target_path)
            backup_path.unlink()
            logger.info("Rolled back: %s", target)
    for target in (created or []):
        target_path = (REPO_ROOT / target).resolve()
        if target_path.exists():
            target_path.unlink()
            logger.info("Removed new file: %s", target)


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [evolution] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # Keep stdout clean for JSON result
    )

    mutations_json = "[]"
    if "--mutations" in sys.argv:
        idx = sys.argv.index("--mutations")
        if idx + 1 < len(sys.argv):
            mutations_json = sys.argv[idx + 1]

    try:
        mutations = json.loads(mutations_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"Invalid JSON: {exc}"}))
        sys.exit(1)

    result = apply_and_validate(mutations)
    print(json.dumps(result))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    _main()
