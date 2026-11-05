"""Structured JSON logging (one JSON object per line) with run-scoped context.

``run_id`` / ``span_id`` are bound through structlog contextvars so every event emitted while a run
is executing is correlated without passing loggers around.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_MAX_FIELD_CHARS = 2000
_REDACT_KEYS = {"api_key", "authorization", "deepseek_api_key", "password", "token"}


def truncate(value: Any, limit: int = _MAX_FIELD_CHARS) -> Any:
    """Keep log lines bounded: long strings / large structures are cut with a marker."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...<{len(value) - limit} more chars>"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, default=str)
        return value if len(text) <= limit else text[:limit] + f"...<{len(text) - limit} more chars>"
    return value


def _redact(_logger, _method, event_dict: dict) -> dict:
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***"
    return event_dict


def configure_logging(level: str = "INFO", log_file: Path | None = None, stream: Any = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(format="%(message)s", level=level.upper(), handlers=handlers, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(default=str),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "harness") -> Any:
    return structlog.get_logger(name)
