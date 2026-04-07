"""WebSocket subscriber registry and sliding-window rate limiting for the streaming API."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Track connected WebSocket clients for live updates and chat.

    Broadcasts iterate a snapshot of connections and drop stale sockets that raise on send.
    """

    def __init__(self) -> None:
        self._live: Set[WebSocket] = set()
        self._chat: Set[WebSocket] = set()

    @property
    def live_count(self) -> int:
        """Number of clients subscribed to the live market stream."""
        return len(self._live)

    @property
    def chat_count(self) -> int:
        """Number of clients connected to advisor chat."""
        return len(self._chat)

    async def connect_live(self, websocket: WebSocket) -> None:
        """Accept the socket and register it for live broadcasts."""
        await websocket.accept()
        self._live.add(websocket)

    async def connect_chat(self, websocket: WebSocket) -> None:
        """Accept the socket and register it for chat responses."""
        await websocket.accept()
        self._chat.add(websocket)

    def disconnect_live(self, websocket: WebSocket) -> None:
        """Remove a live client from the registry."""
        self._live.discard(websocket)

    def disconnect_chat(self, websocket: WebSocket) -> None:
        """Remove a chat client from the registry."""
        self._chat.discard(websocket)

    async def send_json_safe(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """
        Send JSON to one client; return ``False`` if the socket should be dropped.

        Parameters
        ----------
        websocket
            Target connection.
        data
            Serializable dict (FastAPI encodes to JSON).

        Returns
        -------
        bool
            ``True`` if the payload was sent; ``False`` on any send failure.
        """
        try:
            await websocket.send_json(data)
            return True
        except Exception as exc:
            logger.debug("send_json failed: %s", exc)
            return False

    async def broadcast_live(self, data: Dict[str, Any]) -> None:
        """Send ``data`` to every live subscriber, removing dead connections."""
        stale: List[WebSocket] = []
        for ws in list(self._live):
            ok = await self.send_json_safe(ws, data)
            if not ok:
                stale.append(ws)
        for ws in stale:
            self.disconnect_live(ws)


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window limiter: at most ``max_events`` per ``window_seconds`` per key.

    Used for per-IP chat and live inbound message limits.
    """

    def __init__(self, max_events: int, window_seconds: float) -> None:
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """
        Record one event for ``key`` and return whether the limit allows it.

        Returns
        -------
        bool
            ``True`` if under the cap; ``False`` if the window is saturated.
        """
        now = time.monotonic()
        dq = self._events[key]
        cutoff = now - self.window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.max_events:
            return False
        dq.append(now)
        return True
