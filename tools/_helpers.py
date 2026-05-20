"""
Shared IBKR helpers used across tool modules.

Centralizes contract qualification, price snapshots, and account metric
extraction so tool modules stay focused on their domain logic.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ib_insync import IB, Contract, Stock, Ticker

from ibkr_agent.connection import get_connection

logger = logging.getLogger(__name__)


def qualify_us_equity(symbol: str) -> Stock | None:
    """
    Create and qualify a US equity contract. Returns None if IBKR can't
    resolve the symbol (delisted, invalid, etc.).
    """
    ib = get_connection()
    contract = Stock(symbol.upper().strip(), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.warning("Failed to qualify contract for symbol: %s", symbol)
        return None
    return contract


def get_market_price(contract: Contract, timeout_sec: float = 2.0) -> float | None:
    """
    Request a snapshot market price for a qualified contract.
    Returns None if price is unavailable or NaN.

    Always cancels the market data subscription after use to avoid
    exhausting IBKR's concurrent line limit.
    """
    ib = get_connection()
    ticker: Ticker = ib.reqMktData(contract, snapshot=True)
    ib.sleep(timeout_sec)
    price = ticker.marketPrice()
    ib.cancelMktData(contract)

    if price is None or (isinstance(price, float) and math.isnan(price)):
        # Fallback: try last price or close
        price = ticker.last or ticker.close
        if price is None or (isinstance(price, float) and math.isnan(price)):
            logger.warning(
                "No market price available for %s (conId=%s)",
                contract.symbol, contract.conId,
            )
            return None

    return float(price)


def get_account_value(tag: str, currency: str = "USD") -> float | None:
    """
    Extract a single account summary value by tag name.
    Common tags: NetLiquidation, AvailableFunds, BuyingPower,
                 GrossPositionValue, TotalCashValue, MaintMarginReq.
    """
    ib = get_connection()
    for av in ib.accountSummary():
        if av.tag == tag and av.currency == currency:
            try:
                return float(av.value)
            except (ValueError, TypeError):
                return None
    return None


def get_nlv() -> float:
    """
    Net Liquidation Value — the denominator for all risk calculations.
    Raises RuntimeError if unavailable (connection issue, account not loaded).
    """
    nlv = get_account_value("NetLiquidation")
    if nlv is None or nlv <= 0:
        raise RuntimeError(
            "Could not retrieve NetLiquidation from IBKR account summary. "
            "Ensure TWS/Gateway is running and the account is loaded."
        )
    return nlv


def get_gross_exposure() -> float:
    """
    Sum of absolute market values of all open positions.
    Used for total exposure limit checks.
    """
    ib = get_connection()
    total = 0.0
    for pos in ib.positions():
        contract = pos.contract
        ib.qualifyContracts(contract)
        price = get_market_price(contract, timeout_sec=1.0)
        if price is not None:
            total += abs(price * float(pos.position))
    return total


def get_position_for_symbol(symbol: str) -> dict[str, Any] | None:
    """
    Return position details for a specific symbol, or None if no position exists.
    """
    ib = get_connection()
    symbol = symbol.upper().strip()
    for pos in ib.positions():
        if pos.contract.symbol == symbol:
            contract = pos.contract
            ib.qualifyContracts(contract)
            price = get_market_price(contract, timeout_sec=1.0)
            qty = float(pos.position)
            avg_cost = float(pos.avgCost)
            market_value = price * qty if price else None
            unrealized_pl = (price - avg_cost) * qty if price else None
            return {
                "symbol": symbol,
                "contract": contract,
                "qty": qty,
                "avg_cost": round(avg_cost, 4),
                "market_price": round(price, 2) if price else None,
                "market_value": round(market_value, 2) if market_value else None,
                "unrealized_pl": round(unrealized_pl, 2) if unrealized_pl else None,
                "unrealized_pl_pct": (
                    round((price / avg_cost - 1) * 100, 2)
                    if price and avg_cost > 0 else None
                ),
            }
    return None
