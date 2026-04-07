"""
API response cache: Redis when REDIS_URL is set, otherwise in-memory TTL.
Values are pickled for Redis (supports DataFrames, lists, dicts).
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_cache_singleton: Optional["BaseResponseCache"] = None


class BaseResponseCache:
    """Abstract API response cache keyed by string; values may be arbitrary Python objects."""

    backend: str = "memory"

    def get(self, key: str) -> Any:
        """Return a cached value or ``None`` if missing or expired."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store ``value`` with a time-to-live in seconds."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove a key if present."""
        raise NotImplementedError

    def ping(self) -> bool:
        """Return whether the backend is reachable (always ``True`` for in-memory)."""
        return True

    def info(self) -> Dict[str, Any]:
        """Return a small status dict for ``/health``."""
        return {"backend": self.backend, "ok": True}


class InMemoryTTLCache(BaseResponseCache):
    """Process-local TTL cache (no cross-process sharing)."""

    backend = "memory"

    def __init__(self) -> None:
        self._data: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        item = self._data.get(key)
        if not item:
            return None
        exp, val = item
        if time.monotonic() > exp:
            del self._data[key]
            return None
        return val

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._data[key] = (time.monotonic() + max(1, ttl_seconds), value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def info(self) -> Dict[str, Any]:
        return {"backend": self.backend, "ok": True, "entries": len(self._data)}


class RedisResponseCache(BaseResponseCache):
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis  # type: ignore

        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._url = url.split("@")[-1] if "@" in url else url  # hide creds in logs

    def get(self, key: str) -> Any:
        raw = self._client.get(key.encode() if isinstance(key, str) else key)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception as exc:
            logger.warning("Redis cache unpickle failed for %s: %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        bkey = key.encode() if isinstance(key, str) else key
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._client.set(bkey, payload, ex=max(1, ttl_seconds))

    def delete(self, key: str) -> None:
        bkey = key.encode() if isinstance(key, str) else key
        self._client.delete(bkey)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    def info(self) -> Dict[str, Any]:
        ok = self.ping()
        return {"backend": self.backend, "ok": ok, "endpoint": self._url}


def get_response_cache() -> BaseResponseCache:
    """Lazy singleton; prefers Redis if REDIS_URL is set and connection works."""
    global _cache_singleton
    if _cache_singleton is not None:
        return _cache_singleton

    from utils import config

    url = getattr(config, "REDIS_URL", None) or None
    if url:
        try:
            rc = RedisResponseCache(url)
            if rc.ping():
                _cache_singleton = rc
                logger.info("Response cache: Redis connected")
                return _cache_singleton
        except Exception as exc:
            logger.warning("Redis unavailable (%s); using in-memory cache", exc)

    _cache_singleton = InMemoryTTLCache()
    logger.info("Response cache: in-memory TTL")
    return _cache_singleton


def reset_response_cache_for_tests() -> None:
    global _cache_singleton
    _cache_singleton = None
