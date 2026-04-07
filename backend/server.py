"""
FastAPI application: static dashboard, health endpoints, WebSocket live stream and chat.

Run with uvicorn from the project root, for example::

    uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator_agent import OrchestratorAgent
from services.streaming_service import StreamingService
from utils import config
from utils.logging import configure_root_logging
from utils.system_health import get_system_health

configure_root_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Modal Stock Streaming Backend")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

orchestrator = OrchestratorAgent()
streaming = StreamingService(orchestrator)


@app.on_event("startup")
async def startup_event() -> None:
    """Start one asyncio task per configured live ticker (non-blocking across symbols)."""
    asyncio.create_task(streaming.start_stream_workers())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cancel stream workers, then release orchestrator resources (e.g. Neo4j)."""
    await streaming.shutdown()
    orchestrator.close()


@app.get("/")
async def index() -> FileResponse:
    """Serve the single-page streaming dashboard."""
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health() -> JSONResponse:
    """Aggregate health: cache backend, API flags, agent status, streaming configuration."""
    agent_status = orchestrator.get_analysis_status()
    body = get_system_health(agent_status=agent_status)
    body["streaming"] = {
        "tickers": config.LIVE_STREAM_TICKERS,
        "interval_seconds": config.LIVE_STREAM_INTERVAL_SECONDS,
        "max_concurrent_pipelines": config.LIVE_STREAM_MAX_CONCURRENT,
        "max_tickers": config.LIVE_STREAM_MAX_TICKERS,
        "websocket": {
            "heartbeat_seconds": config.WS_HEARTBEAT_INTERVAL_SECONDS,
            "chat_rate_limit_per_minute": config.WS_CHAT_RATE_LIMIT_PER_MINUTE,
        },
    }
    return JSONResponse(body)


@app.get("/health/system")
async def health_system() -> JSONResponse:
    """System health without streaming-specific metadata."""
    agent_status = orchestrator.get_analysis_status()
    return JSONResponse(get_system_health(agent_status=agent_status))


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Bidirectional WebSocket: live quotes, deltas, compact ``stream_tick``, optional subscribe."""
    await streaming.handle_live_ws(websocket)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Advisor chat with token streaming (Gemini) and rate limiting per client IP."""
    await streaming.handle_chat_ws(websocket)
