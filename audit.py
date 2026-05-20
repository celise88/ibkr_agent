"""
Structured audit logging.

Every tool invocation, trade submission, and risk rejection is logged to JSONL
files for post-hoc analysis. This is the primary debugging tool when the agent
does something unexpected.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ibkr_agent.config import LOG

logger = logging.getLogger(__name__)


def _ensure_log_dir() -> Path:
    LOG.log_dir.mkdir(parents=True, exist_ok=True)
    return LOG.log_dir


def _serialize(obj: Any) -> Any:
    """Make arbitrary objects JSON-safe."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def log_event(log_file: str, event_type: str, payload: dict[str, Any]) -> None:
    """
    Append a structured event to a JSONL audit log.

    Args:
        log_file: Filename within the log directory (e.g., "trades.jsonl").
        event_type: Category tag (e.g., "order_submitted", "risk_rejected").
        payload: Arbitrary dict of event data.
    """
    path = _ensure_log_dir() / log_file
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_serialize) + "\n")
    except OSError as exc:
        logger.error("Failed to write audit log to %s: %s", path, exc)


def log_trade(event_type: str, payload: dict[str, Any]) -> None:
    """Convenience wrapper: log to the trade audit file."""
    log_event(LOG.trade_log, event_type, payload)


def log_agent(event_type: str, payload: dict[str, Any]) -> None:
    """Convenience wrapper: log to the agent decision file."""
    log_event(LOG.agent_log, event_type, payload)


def setup_logging() -> None:
    """
    Configure stdlib logging with structured format.
    Call once at startup (main.py).
    """
    _ensure_log_dir()

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG.level, logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    # File handler for full debug output
    file_handler = logging.FileHandler(
        LOG.log_dir / "agent_debug.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s"
        )
    )
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
