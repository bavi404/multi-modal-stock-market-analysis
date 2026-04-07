"""
Central logging configuration for CLI scripts, evaluation runners, and the API server.

All modules should use ``logging.getLogger(__name__)`` (or :func:`get_logger`) so messages
share one format once the root logger is configured.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional

# Shared layout: timestamp | level | logger name | message (stable for grep and log aggregation)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Prefer ``logging.getLogger(__name__)`` in application code;
    this alias documents intent for libraries.
    """
    return logging.getLogger(name)


def configure_root_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Apply a single Formatter to the root logger and replace existing handlers.

    Use for long-running processes (e.g. FastAPI) where ``basicConfig`` would be inconsistent.

    Parameters
    ----------
    level
        Default level when ``verbose`` is False.
    log_file
        If set, append a UTF-8 :class:`~logging.FileHandler` with the same format.
    verbose
        When True, root and handlers use DEBUG.
    """
    root = logging.getLogger()
    effective = logging.DEBUG if verbose else level
    root.setLevel(effective)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(effective)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(effective)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for command-line tools: console plus a daily log file in the CWD.

    Parameters
    ----------
    verbose
        Enable DEBUG on the root logger and handlers.
    """
    log_path = f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.log"
    configure_root_logging(verbose=verbose, log_file=log_path, level=logging.INFO)
