"""
Structured WebSocket payloads (JSON).

Every server message uses a versioned envelope: ``schema_version``, ``type``, monotonic ``seq``,
``ts`` (UTC ISO 8601), and a ``payload`` object. Clients may rely on ``type`` for dispatch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class WSEnvelope(BaseModel):
    """Wire format for outbound WebSocket messages."""

    schema_version: int = SCHEMA_VERSION
    type: str
    seq: int
    ts: str
    payload: Dict[str, Any] = Field(default_factory=dict)


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with timezone offset."""
    return datetime.now(timezone.utc).isoformat()


_seq_counter = 0


def next_seq() -> int:
    """Return the next monotonic sequence number for envelope ordering."""
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


def envelope(
    msg_type: str,
    payload: Optional[Dict[str, Any]] = None,
    seq: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable envelope dict.

    Parameters
    ----------
    msg_type
        Short string discriminator (e.g. ``live_delta``, ``stream_tick``).
    payload
        Domain payload; omitted keys default to an empty object.
    seq
        Optional explicit sequence; normally assigned by :func:`next_seq`.
    """
    return WSEnvelope(
        type=msg_type,
        seq=seq if seq is not None else next_seq(),
        ts=utc_now_iso(),
        payload=payload or {},
    ).model_dump(mode="json")


class LiveClientMessage(BaseModel):
    """Subset of client messages accepted on ``/ws/live``."""

    type: Literal["pong", "ping", "subscribe"]
    payload: Dict[str, Any] = Field(default_factory=dict)


class ChatClientMessage(BaseModel):
    """Expected shape for advisor chat requests."""

    type: Literal["chat_message"] = "chat_message"
    payload: Dict[str, Any] = Field(default_factory=dict)


def parse_chat_inbound(data: Dict[str, Any]) -> tuple[str, str]:
    """
    Extract ``(user_text, ticker)`` from flexible client JSON (with or without envelope).

    Returns
    -------
    tuple[str, str]
        Message body and uppercased ticker (defaults to ``AAPL`` if missing).
    """
    if data.get("type") == "chat_message" and isinstance(data.get("payload"), dict):
        payload = data["payload"]
        return (str(payload.get("message") or "").strip(), str(payload.get("ticker") or "AAPL").upper())
    return (str(data.get("message") or "").strip(), str(data.get("ticker") or "AAPL").upper())


def compute_live_delta(prev: Dict[str, Any], curr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return field-level changes from ``prev`` to ``curr``, or ``None`` if identical.

    The ``type`` field is ignored for comparison.
    """
    changes: Dict[str, Any] = {}
    for key, value in curr.items():
        if key == "type":
            continue
        if prev.get(key) != value:
            changes[key] = value
    return changes if changes else None
