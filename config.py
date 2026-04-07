"""
Legacy shim: ``import config`` re-exports :mod:`utils.config`.

Prefer ``from utils import config`` or ``from utils.config import TWITTER_BEARER_TOKEN``.
"""
from utils.config import *  # noqa: F403
