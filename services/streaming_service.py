"""
Live WebSocket broadcasting and Gemini-backed advisor chat.

The live path runs one asyncio task per ticker: each loop fetches a :class:`~models.data_models.LiveUpdateResult`
through the orchestrator, broadcasts full snapshots/deltas for the UI, and emits a compact
``stream_tick`` payload (price, prediction, sentiment). A semaphore caps concurrent pipeline runs
across tickers so external APIs are not overwhelmed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

from utils import config
from agents.orchestrator_agent import OrchestratorAgent
from models.data_models import ChatTurn
from .websocket_manager import ConnectionManager, SlidingWindowRateLimiter
from .ws_messages import compute_live_delta, envelope, parse_chat_inbound

logger = logging.getLogger(__name__)


class StreamingService:
    """
    Owns WebSocket connection pools, rate limits, live ticker state, and chat memory.
    Keeps HTTP route handlers thin.
    """

    def __init__(self, orchestrator: OrchestratorAgent) -> None:
        self._orchestrator = orchestrator
        self.manager = ConnectionManager()
        self._last_live_by_ticker: Dict[str, Dict[str, Any]] = {}
        self._advisor_chat_memory: Dict[int, deque] = {}
        self._extra_tickers: Set[str] = set()
        self._stop = asyncio.Event()
        self._worker_sem = asyncio.Semaphore(max(1, config.LIVE_STREAM_MAX_CONCURRENT))
        self._ticker_tasks: Dict[str, asyncio.Task] = {}
        self._worker_lock = asyncio.Lock()
        self.chat_rate = SlidingWindowRateLimiter(
            max_events=max(1, config.WS_CHAT_RATE_LIMIT_PER_MINUTE),
            window_seconds=60.0,
        )
        self.live_inbound_rate = SlidingWindowRateLimiter(
            max_events=max(1, config.WS_LIVE_INBOUND_RATE_LIMIT_PER_MINUTE),
            window_seconds=60.0,
        )

    @staticmethod
    def _normalize_live_snapshot(snap: Any) -> Dict[str, Any]:
        """Convert a Pydantic live snapshot to a plain dict for delta comparison."""
        row = snap.model_dump(mode="json")
        row.pop("type", None)
        return row

    @staticmethod
    def _compact_stream_tick(ticker: str, snap: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal per-ticker row for real-time consumers."""
        return {
            "ticker": ticker,
            "price": snap.get("latest_price"),
            "prediction": snap.get("predicted_price"),
            "sentiment": snap.get("sentiment_score"),
        }

    def _all_tickers(self) -> List[str]:
        """Configured tickers plus WebSocket ``subscribe`` additions, deduplicated and capped."""
        seen: Set[str] = set()
        out: List[str] = []
        for t in list(config.LIVE_STREAM_TICKERS) + list(self._extra_tickers):
            u = str(t).strip().upper()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out[: max(1, config.LIVE_STREAM_MAX_TICKERS)]

    async def _ensure_ticker_worker(self, ticker: str) -> None:
        """One asyncio task per ticker; tickers do not block each other."""
        async with self._worker_lock:
            existing = self._ticker_tasks.get(ticker)
            if existing is not None and not existing.done():
                return
            self._ticker_tasks[ticker] = asyncio.create_task(
                self._ticker_stream_loop(ticker),
                name=f"live_stream:{ticker}",
            )

    async def _ticker_stream_loop(self, ticker: str) -> None:
        interval = max(5, config.LIVE_STREAM_INTERVAL_SECONDS)
        while not self._stop.is_set():
            try:
                async with self._worker_sem:
                    payload = await self._orchestrator.get_live_snapshot_async(ticker)
                curr = self._normalize_live_snapshot(payload)
                compact = self._compact_stream_tick(ticker, curr)
                prev = self._last_live_by_ticker.get(ticker)
                if prev is None:
                    self._last_live_by_ticker[ticker] = curr
                    await self.manager.broadcast_live(
                        envelope("live_snapshot", {"ticker": ticker, "snapshot": curr})
                    )
                    await self.manager.broadcast_live(envelope("stream_tick", compact))
                else:
                    changes = compute_live_delta(prev, curr)
                    if changes:
                        self._last_live_by_ticker[ticker] = curr
                        await self.manager.broadcast_live(
                            envelope("live_delta", {"ticker": ticker, "changes": changes})
                        )
                        await self.manager.broadcast_live(envelope("stream_tick", compact))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Live stream ticker %s error: %s", ticker, exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                continue

    async def start_stream_workers(self) -> None:
        """Spawn concurrent per-ticker loops (non-blocking across tickers)."""
        for ticker in self._all_tickers():
            await self._ensure_ticker_worker(ticker)

    async def shutdown(self) -> None:
        """Stop all ticker workers (call before closing orchestrator)."""
        self._stop.set()
        async with self._worker_lock:
            tasks = list(self._ticker_tasks.values())
            self._ticker_tasks.clear()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _advisor_history(self, websocket: WebSocket) -> deque:
        """Per-connection deque of :class:`~models.data_models.ChatTurn` for advisor context."""
        return self._advisor_chat_memory.setdefault(
            id(websocket),
            deque(maxlen=max(2, config.ADVISOR_HISTORY_MAX_TURNS * 2)),
        )

    @staticmethod
    def _client_ip(ws: WebSocket) -> str:
        """Best-effort client IP for rate limiting (may be a proxy address)."""
        if ws.client:
            return ws.client.host or "unknown"
        return "unknown"

    async def _subscribe_tickers(self, tickers: List[Any]) -> None:
        """Add tickers at runtime (capped); each gets its own concurrent worker."""
        try:
            for raw in tickers:
                u = str(raw).strip().upper()
                if not u:
                    continue
                cur = set(self._all_tickers())
                if u not in cur and len(cur) >= config.LIVE_STREAM_MAX_TICKERS:
                    logger.warning(
                        "stream subscribe: max tickers (%s) reached, ignoring %s",
                        config.LIVE_STREAM_MAX_TICKERS,
                        u,
                    )
                    continue
                self._extra_tickers.add(u)
                await self._ensure_ticker_worker(u)
        except Exception as exc:
            logger.exception("subscribe tickers error: %s", exc)

    async def _send_live_catchup(self, websocket: WebSocket) -> None:
        """On connect, replay last known snapshots and compact ticks for all tracked symbols."""
        for ticker in sorted(self._last_live_by_ticker.keys()):
            snap = self._last_live_by_ticker[ticker]
            await self.manager.send_json_safe(
                websocket,
                envelope("live_snapshot", {"ticker": ticker, "snapshot": snap}),
            )
            await self.manager.send_json_safe(
                websocket,
                envelope("stream_tick", self._compact_stream_tick(ticker, snap)),
            )

    async def _live_heartbeat(self, websocket: WebSocket, stop: asyncio.Event) -> None:
        """Periodic ``ping`` frames until ``stop`` is set (keeps proxies from closing idle sockets)."""
        interval = max(5, config.WS_HEARTBEAT_INTERVAL_SECONDS)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                await self.manager.send_json_safe(websocket, envelope("ping", {}))

    async def _handle_live_inbound(self, websocket: WebSocket, raw: str, ip: str) -> None:
        """Handle ping/pong and optional ``subscribe`` ticker list (rate-limited per IP)."""
        if not self.live_inbound_rate.allow(ip):
            return
        raw = raw.strip()
        if not raw:
            return
        if raw.lower() == "ping" or raw == '{"type":"ping"}':
            await self.manager.send_json_safe(websocket, envelope("pong", {}))
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_type = data.get("type")
        if msg_type == "pong":
            return
        if msg_type == "ping":
            await self.manager.send_json_safe(websocket, envelope("pong", {}))
            return
        if msg_type == "subscribe":
            tickers = data.get("tickers")
            if tickers is None and isinstance(data.get("payload"), dict):
                tickers = data["payload"].get("tickers")
            if isinstance(tickers, list):
                asyncio.create_task(self._subscribe_tickers(tickers))
            return

    async def handle_live_ws(self, websocket: WebSocket) -> None:
        """WebSocket lifecycle for ``/ws/live``: connect, catch-up, heartbeat, inbound loop."""
        await self.manager.connect_live(websocket)
        ip = self._client_ip(websocket)
        stop = asyncio.Event()
        hb = asyncio.create_task(self._live_heartbeat(websocket, stop))
        try:
            await self.manager.send_json_safe(
                websocket,
                envelope(
                    "live_connected",
                    {
                        "tickers": self._all_tickers(),
                        "heartbeat_seconds": config.WS_HEARTBEAT_INTERVAL_SECONDS,
                        "max_concurrent": config.LIVE_STREAM_MAX_CONCURRENT,
                    },
                ),
            )
            await self._send_live_catchup(websocket)
            while True:
                raw = await websocket.receive_text()
                await self._handle_live_inbound(websocket, raw, ip)
        except WebSocketDisconnect:
            logger.debug("Live WS disconnected: %s", ip)
        except Exception as exc:
            logger.debug("Live WS closed: %s", exc)
        finally:
            stop.set()
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
            self.manager.disconnect_live(websocket)

    async def handle_chat_ws(self, websocket: WebSocket) -> None:
        """WebSocket lifecycle for ``/ws/chat``: stream Gemini tokens with bounded chat memory."""
        await self.manager.connect_chat(websocket)
        ip = self._client_ip(websocket)
        stop = asyncio.Event()
        hb = asyncio.create_task(self._live_heartbeat(websocket, stop))
        try:
            await self.manager.send_json_safe(
                websocket,
                envelope("chat_connected", {"heartbeat_seconds": config.WS_HEARTBEAT_INTERVAL_SECONDS}),
            )
            while True:
                msg = await websocket.receive_json()
                user_text, ticker = parse_chat_inbound(msg if isinstance(msg, dict) else {})
                if not user_text:
                    await self.manager.send_json_safe(
                        websocket,
                        envelope("error", {"message": "Empty message", "code": "empty_message"}),
                    )
                    continue
                if not self.chat_rate.allow(ip):
                    await self.manager.send_json_safe(
                        websocket,
                        envelope("error", {"message": "Rate limit exceeded", "code": "rate_limited"}),
                    )
                    continue

                prior = list(self._advisor_history(websocket))
                data_layer = await self._orchestrator.get_advisor_data_layer_async(ticker)
                stream_id = str(uuid.uuid4())
                await self.manager.send_json_safe(websocket, envelope("chat_start", {"stream_id": stream_id}))

                try:
                    chunks: List[str] = []
                    for token_text in self._orchestrator.advisor_agent.stream_advice_tokens(
                        user_text, data_layer, prior
                    ):
                        chunks.append(token_text)
                        await self.manager.send_json_safe(
                            websocket,
                            envelope("chat_token", {"stream_id": stream_id, "token": token_text}),
                        )
                    await self.manager.send_json_safe(websocket, envelope("chat_end", {"stream_id": stream_id}))
                    assistant_text = "".join(chunks)
                    max_mem = int(getattr(config, "ADVISOR_ASSISTANT_MEMORY_MAX_CHARS", 4000))
                    if len(assistant_text) > max_mem:
                        assistant_text = assistant_text[:max_mem]
                    hist = self._advisor_history(websocket)
                    hist.append(ChatTurn(role="user", content=user_text))
                    hist.append(ChatTurn(role="assistant", content=assistant_text))
                except Exception as exc:
                    logger.exception("Chat stream error: %s", exc)
                    await self.manager.send_json_safe(
                        websocket,
                        envelope(
                            "error",
                            {"message": f"Advisor error: {exc}", "stream_id": stream_id, "code": "advisor_error"},
                        ),
                    )
                    await self.manager.send_json_safe(websocket, envelope("chat_end", {"stream_id": stream_id}))
        except WebSocketDisconnect:
            logger.debug("Chat WS disconnected: %s", ip)
        except Exception as exc:
            logger.debug("Chat WS closed: %s", exc)
        finally:
            stop.set()
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
            self.manager.disconnect_chat(websocket)
            self._advisor_chat_memory.pop(id(websocket), None)
