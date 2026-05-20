"""
Portfolio snapshot tool.

Returns a structured view of account state and all open positions.
The agent calls this before any trade decision.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from ibkr_agent.audit import log_agent
from ibkr_agent.connection import get_connection, ensure_connected
from ibkr_agent.tools._helpers import get_market_price, get_nlv

logger = logging.getLogger(__name__)


@tool
@ensure_connected
def get_portfolio_snapshot() -> dict:
    """
    Retrieve current account state and all open positions.

    Returns:
        dict with keys:
        - account: NLV, cash, buying power, margin usage
        - positions: list of open positions with unrealized P&L
        - summary: aggregate exposure and P&L metrics
    """
    ib = get_connection()

    # ------------------------------------------------------------------
    # Account-level metrics
    # ------------------------------------------------------------------
    target_tags = {
        "NetLiquidation",
        "TotalCashValue",
        "AvailableFunds",
        "BuyingPower",
        "GrossPositionValue",
        "MaintMarginReq",
        "InitMarginReq",
    }
    account = {}
    for av in ib.accountSummary():
        if av.tag in target_tags and av.currency == "USD":
            try:
                account[av.tag] = round(float(av.value), 2)
            except (ValueError, TypeError):
                account[av.tag] = av.value

    nlv = account.get("NetLiquidation", 0.0)

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------
    positions = []
    total_long_value = 0.0
    total_short_value = 0.0
    total_unrealized_pl = 0.0

    for pos in ib.positions():
        contract = pos.contract
        ib.qualifyContracts(contract)

        qty = float(pos.position)
        avg_cost = float(pos.avgCost)
        price = get_market_price(contract, timeout_sec=1.5)

        if price is not None:
            market_value = price * qty
            unrealized_pl = (price - avg_cost) * qty
            unrealized_pl_pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0

            if qty > 0:
                total_long_value += market_value
            else:
                total_short_value += abs(market_value)
            total_unrealized_pl += unrealized_pl
        else:
            market_value = None
            unrealized_pl = None
            unrealized_pl_pct = None

        positions.append({
            "symbol": contract.symbol,
            "sec_type": contract.secType,
            "exchange": contract.primaryExchange or contract.exchange,
            "qty": qty,
            "side": "LONG" if qty > 0 else "SHORT",
            "avg_cost": round(avg_cost, 4),
            "market_price": round(price, 2) if price is not None else "unavailable",
            "market_value": round(market_value, 2) if market_value is not None else "unavailable",
            "unrealized_pl": round(unrealized_pl, 2) if unrealized_pl is not None else "unavailable",
            "unrealized_pl_pct": round(unrealized_pl_pct, 2) if unrealized_pl_pct is not None else "unavailable",
            "weight_pct": round(abs(market_value) / nlv * 100, 2) if market_value and nlv > 0 else "unavailable",
        })

    # ------------------------------------------------------------------
    # Summary metrics for the agent
    # ------------------------------------------------------------------
    gross_exposure = total_long_value + total_short_value
    summary = {
        "total_positions": len(positions),
        "total_long_value": round(total_long_value, 2),
        "total_short_value": round(total_short_value, 2),
        "gross_exposure": round(gross_exposure, 2),
        "gross_exposure_pct": round(gross_exposure / nlv * 100, 2) if nlv > 0 else 0.0,
        "net_exposure": round(total_long_value - total_short_value, 2),
        "total_unrealized_pl": round(total_unrealized_pl, 2),
    }

    result = {
        "account": account,
        "positions": positions,
        "summary": summary,
    }

    log_agent("portfolio_snapshot", {
        "nlv": nlv,
        "position_count": len(positions),
        "gross_exposure_pct": summary["gross_exposure_pct"],
        "total_unrealized_pl": summary["total_unrealized_pl"],
    })

    return result
