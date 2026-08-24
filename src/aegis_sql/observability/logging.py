"""Structured logging with automatic trace-id propagation."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from aegis_sql.observability.trace import current_trace_id

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        trace_id = current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        trace_id = current_trace_id()
        prefix = f"[{trace_id}] " if trace_id else ""
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} {prefix}{record.name}: {record.getMessage()}"
        extra = getattr(record, "extra_fields", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    global _CONFIGURED
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root = logging.getLogger("aegis_sql")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return BoundLogger(logging.getLogger(f"aegis_sql.{name}"))


class BoundLogger:
    """Thin wrapper giving ``log.info("msg", key=value)`` ergonomics."""

    __slots__ = ("_log",)

    def __init__(self, log: logging.Logger) -> None:
        self._log = log

    def _emit(self, level: int, msg: str, exc_info: bool = False, **fields: Any) -> None:
        self._log.log(level, msg, exc_info=exc_info, extra={"extra_fields": fields})

    def debug(self, msg: str, **f: Any) -> None:
        self._emit(logging.DEBUG, msg, **f)

    def info(self, msg: str, **f: Any) -> None:
        self._emit(logging.INFO, msg, **f)

    def warning(self, msg: str, **f: Any) -> None:
        self._emit(logging.WARNING, msg, **f)

    def error(self, msg: str, exc_info: bool = False, **f: Any) -> None:
        self._emit(logging.ERROR, msg, exc_info=exc_info, **f)
