"""
Configuration: risk limits, connection parameters, and environment loading.

Risk limits are frozen at import time. The LLM cannot modify them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}. "
            f"Set it in your .env file or shell environment."
        )
    return val


# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IBKRConnectionConfig:
    """TWS/Gateway connection parameters."""

    host: str = "127.0.0.1"
    port: int = 4002                # IB Gateway paper default; TWS paper = 7497
    client_id: int = 1
    timeout: float = 15.0
    readonly: bool = False
    market_data_type: int = 3       # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen


# ---------------------------------------------------------------------------
# Risk limits (hard-coded, immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """
    Programmatic risk guardrails enforced at the execution layer.
    These are NOT suggestions to the LLM — they are hard constraints
    that reject orders before they reach IBKR.
    """

    # Position sizing
    max_position_pct: float = 0.05          # Max 5% of NLV in a single name
    max_total_exposure_pct: float = 0.60    # Max 60% of NLV across all positions
    min_order_notional: float = 100.0       # Reject dust orders below $100

    # Loss limits
    max_loss_per_trade_pct: float = 0.02    # 2% hard stop per position
    max_daily_loss_pct: float = 0.05        # 5% daily drawdown circuit breaker

    # Activity limits
    max_daily_trades: int = 20              # Circuit breaker on trade count

    # Universe restrictions
    allowed_sec_types: frozenset[str] = frozenset({"STK"})
    allowed_exchanges: frozenset[str] = frozenset({"SMART", "NYSE", "NASDAQ", "ARCA", "BATS"})
    allowed_currencies: frozenset[str] = frozenset({"USD"})


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Logging / audit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogConfig:
    log_dir: Path = Path("logs")
    trade_log: str = "trades.jsonl"
    agent_log: str = "agent.jsonl"
    level: str = "INFO"


# ---------------------------------------------------------------------------
# Singleton instances (import these directly)
# ---------------------------------------------------------------------------

IBKR_CONFIG = IBKRConnectionConfig(
    host=os.environ.get("IBKR_HOST", "127.0.0.1"),
    port=int(os.environ.get("IBKR_PORT", "4002")),
    client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
    market_data_type=int(os.environ.get("IBKR_MARKET_DATA_TYPE", "3")),
)

RISK = RiskConfig()
LLM = LLMConfig()
LOG = LogConfig()

# Anthropic API key — validated at import so failures are loud and early
ANTHROPIC_API_KEY = _require_env("ANTHROPIC_API_KEY")
