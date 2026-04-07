"""IPC protocol — JSON messages over Unix domain sockets."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single IPC message between supervisor and worker."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = "request"  # request | progress | result | error | control
    method: str = ""       # run | cancel | status | health | shutdown | evolve | ready
    params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, raw: str) -> Message:
        d = json.loads(raw)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ── Convenience constructors ──

    @classmethod
    def request(cls, method: str, **params: Any) -> Message:
        return cls(type="request", method=method, params=params)

    @classmethod
    def progress(cls, msg_id: str, phase: str, detail: str = "") -> Message:
        return cls(id=msg_id, type="progress", data={"phase": phase, "detail": detail})

    @classmethod
    def result(cls, msg_id: str, data: dict[str, Any]) -> Message:
        return cls(id=msg_id, type="result", data=data)

    @classmethod
    def error(cls, msg_id: str, error: str) -> Message:
        return cls(id=msg_id, type="error", error=error)

    @classmethod
    def control(cls, method: str, **data: Any) -> Message:
        return cls(type="control", method=method, data=data)


async def send_message(writer: asyncio.StreamWriter, msg: Message) -> None:
    """Send a single message as a newline-delimited JSON line."""
    line = msg.to_json() + "\n"
    writer.write(line.encode())
    await writer.drain()


async def recv_message(reader: asyncio.StreamReader) -> Message | None:
    """Read a single message. Returns None on EOF."""
    try:
        line = await reader.readline()
    except (ConnectionResetError, asyncio.IncompleteReadError):
        return None
    if not line:
        return None
    return Message.from_json(line.decode().strip())


async def recv_message_timeout(
    reader: asyncio.StreamReader, timeout: float
) -> Message | None:
    """Read a message with timeout. Returns None on timeout or EOF."""
    try:
        return await asyncio.wait_for(recv_message(reader), timeout=timeout)
    except asyncio.TimeoutError:
        return None
