"""Aggregated health snapshot for APIs and operations (cache, data sources, agents)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils import config
from utils.response_cache import get_response_cache


def get_system_health(agent_status: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Build a JSON-serializable health document for load balancers and dashboards.

    Parameters
    ----------
    agent_status
        Optional map of agent name to human-readable status (from the orchestrator).

    Returns
    -------
    dict
        Status string, cache info, TTL hints, configured data source flags, and ``agents``.
    """
    cache = get_response_cache()
    cache_info = cache.info()
    overall = "healthy"
    if cache.backend == "redis" and not cache_info.get("ok"):
        overall = "degraded"

    return {
        "status": overall,
        "ok": overall == "healthy",
        "cache": cache_info,
        "ttl_seconds": {
            "stock": int(getattr(config, "CACHE_TTL_STOCK_SECONDS", 300)),
            "stocktwits": int(getattr(config, "CACHE_TTL_STOCKTWITS_SECONDS", 120)),
            "reddit": int(getattr(config, "CACHE_TTL_REDDIT_SECONDS", 180)),
            "news": int(getattr(config, "CACHE_TTL_NEWS_SECONDS", 600)),
        },
        "data_sources": {
            "stocktwits_configured": True,
            "reddit_configured": bool(config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET)
            or bool(getattr(config, "REDDIT_DATASET_PATH", None)),
            "news_configured": bool(config.NEWS_API_KEY),
            "yfinance": True,
        },
        "redis_url_configured": bool(getattr(config, "REDIS_URL", None)),
        "gemini_configured": bool(getattr(config, "ADVISOR_GEMINI_API_KEY", None) or config.GEMINI_API_KEY),
        "neo4j_configured": bool(config.NEO4J_PASSWORD),
        "agents": agent_status or {},
    }
