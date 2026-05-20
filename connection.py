"""
IBKR connection manager.

Maintains a module-level singleton IB instance with automatic reconnection.
ib_insync runs its own event loop internally; the ib.sleep() calls in tools
yield to that loop so pending responses can be processed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator
from functools import wraps
from ib_insync import IB
import asyncio

from ibkr_agent.config import IBKR_CONFIG

logger = logging.getLogger(__name__)

# Module-level singleton
_ib: IB | None = None


def _ensure_event_loop() -> None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def get_connection() -> IB:
    """
    Return a connected IB instance. Reconnects transparently if the socket
    has dropped (common with TWS/Gateway idle timeouts).

    Thread safety: ib_insync is NOT thread-safe. All calls must happen on the
    same thread that owns the event loop. This is fine for LangGraph's default
    synchronous executor but would need asyncio adaptation for concurrent use.
    """
    global _ib

    _ensure_event_loop()

    if _ib is None:
        _ib = IB()

    if not _ib.isConnected():
        logger.info(
            "Connecting to IBKR at %s:%d (clientId=%d, readonly=%s)",
            IBKR_CONFIG.host,
            IBKR_CONFIG.port,
            IBKR_CONFIG.client_id,
            IBKR_CONFIG.readonly,
        )
        _ib.connect(
            host=IBKR_CONFIG.host,
            port=IBKR_CONFIG.port,
            clientId=IBKR_CONFIG.client_id,
            timeout=IBKR_CONFIG.timeout,
            readonly=IBKR_CONFIG.readonly,
        )
        _ib.reqMarketDataType(IBKR_CONFIG.market_data_type)
        logger.info(
            "Connected. Market data type set to %d.", IBKR_CONFIG.market_data_type
        )

    return _ib


def disconnect() -> None:
    """Cleanly disconnect and reset the singleton."""
    global _ib
    if _ib is not None and _ib.isConnected():
        logger.info("Disconnecting from IBKR.")
        _ib.disconnect()
    _ib = None


@contextmanager
def managed_connection() -> Generator[IB, None, None]:
    """
    Context manager for scripts that want automatic cleanup:

        with managed_connection() as ib:
            ...
    """
    ib = get_connection()
    try:
        yield ib
    finally:
        disconnect()


def ensure_connected(func):
    """
    Decorator that guarantees a live connection before calling the wrapped
    function. Retries once on ConnectionError (handles TWS restarts / brief
    network blips).
    """
    def wrapper(*args, **kwargs):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                get_connection()
                return func(*args, **kwargs)
            except (ConnectionError, OSError, TimeoutError) as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Connection lost (%s), reconnecting (attempt %d/%d)...",
                        exc, attempt + 1, max_retries,
                    )
                    disconnect()
                    time.sleep(2)
                else:
                    raise

    wrapper = wraps(func)(wrapper)
    return wrapper
