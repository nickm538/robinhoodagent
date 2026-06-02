"""Centralised logging using rich when available, plain logging otherwise."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("RH_LOG_LEVEL", "INFO")).upper()
    try:
        from rich.logging import RichHandler  # type: ignore

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich optional
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    logging.basicConfig(level=lvl, format=fmt, handlers=[handler], force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
