"""Centralised logging setup + request-id helper.

Replaces the sample's mix of ``print`` and the recurring ``datetime.now`` (missing
parentheses) bug with a single configured logger.
"""

import logging
import sys
from uuid import uuid4

from settings import LOG_LEVEL

_CONFIGURED = False


def configureLogging() -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    _CONFIGURED = True


def getLogger(name: str) -> logging.Logger:
    configureLogging()
    return logging.getLogger(name)


def newRequestID() -> str:
    """A short, unique id to correlate a request across log lines."""
    return f"req-{uuid4().hex[:16]}"
