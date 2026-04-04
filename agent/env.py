"""Load .env file from project root into os.environ."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv() -> None:
    """Load .env variables into os.environ (does not overwrite existing vars)."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
