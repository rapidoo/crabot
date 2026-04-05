"""Async embedding wrapper using the ollama SDK."""

from __future__ import annotations

import asyncio
import logging

from agent.config import get_settings

logger = logging.getLogger(__name__)

# ollama SDK is an optional dependency
try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    _ollama = None  # type: ignore


class Embedder:
    """Generate embeddings via the ollama SDK (async-safe)."""

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self._model = model or settings.memory.embedding_model
        self._available = HAS_OLLAMA

    @property
    def available(self) -> bool:
        return self._available

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text.

        Returns an empty list if ollama is not installed or the call fails.
        """
        if not self._available or not text.strip():
            return []

        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: _ollama.embed(
                    model=self._model,
                    input=text[:2000],
                ),
            )
            return resp.get("embeddings", [[]])[0]
        except (AttributeError, TypeError):
            # Fallback for older ollama SDK versions
            try:
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: _ollama.embeddings(
                        model=self._model,
                        prompt=text[:2000],
                    ),
                )
                return resp.get("embedding", [])
            except Exception as exc:
                logger.warning("Embedding failed (legacy API): %s", exc)
                return []
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return []
