"""
Shared base class for agents that wrap blocking libraries behind asyncio-friendly helpers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class BaseAgent:
    """
    Minimal base: named logger and ``asyncio.to_thread`` wrapper for blocking callables.

    Subclasses typically own model pipelines or I/O clients and expose ``*_async`` methods.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_blocking(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run a blocking function in a worker thread so the event loop stays responsive.

        Parameters
        ----------
        fn
            Synchronous callable (e.g. Hugging Face ``pipeline`` or pandas I/O).
        *args, **kwargs
            Forwarded to ``fn``.

        Returns
        -------
        Whatever ``fn`` returns.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

    def log_stage(self, message: str, *args: Any) -> None:
        """Structured info log line (``%``-style formatting supported)."""
        self.logger.info(message, *args)
