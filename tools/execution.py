"""
Trade execution tools with hard-coded risk guardrails.

Every order passes through a deterministic risk gate before reaching IBKR.
The LLM cannot bypass, negotiate, or reinterpret these limits.

Provides three tools:
  - place_trade: single market or limit order
  - place_bracket_trade: atomic entry + take-profit + stop-loss
  - close_position: liquidate an existing position by symbol
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ib_insync import (
    IB,
    LimitOrder,
    MarketOrder,
    StopOrder,
)
from langchain_core.tools import tool

from ibkr_agent.audit import log_trade
from ibkr_agent.config import RISK
from ibkr_agent.connection import get_connection, ensure_connected
from ibkr_agent.tools._helpers import (
    get_gross_exposure,
    get_market_price,
    get_nlv,
    get_position_for_symbol,
    qualify_us_equity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk validation (deterministic, non-negotiable)
# ---------------------------------------------------------------------------

def _validate_trade(
    symbol: str,
    side: str,
    quantity: int,
    estimated_price: float,
    nlv: float,
) -> dict | None:
    """
    Run all risk checks. Returns a rejection dict if any check fails,
    or None if the trade is clear.
    """
    notional = estimated_price * quantity

    # --- Position size limit ---
    max_notional = nlv * RISK.max_position_pct
    if notional > max_notional:
        max_shares = int(max_notional / estimated_price)
        return {
            "status": "REJECTED",
            "rule": "max_position_pct",
            "reason": (
                f"Notional ${notional:,.0f} ({quantity} × ${estimated_price:.2f}) "
                f"exceeds {RISK.max_position_pct:.0%} of NLV (${nlv:,.0f}). "
                f"Maximum: {max_shares} shares (${max_notional:,.0f})."
            ),
        }

    # --- Minimum order size ---
    if notional < RISK.min_order_notional:
        return {
            "status": "REJECTED",
            "rule": "min_order_notional",
            "reason": (
                f"Order notional ${notional:,.0f} is below the "
                f"${RISK.min_order_notional:,.0f} minimum."
            ),
        }

    # --- Total exposure limit (buys only) ---
    if side == "BUY":
        current_exposure = get_gross_exposure()
        projected = current_exposure + notional
        max_exposure = nlv * RISK.max_total_exposure_pct
        if projected > max_exposure:
            headroom = max(0, max_exposure - current_exposure)
            return {
                "status": "REJECTED",
                "rule": "max_total_exposure_pct",
                "reason": (
                    f"Adding ${notional:,.0f} would bring gross exposure to "
                    f"${projected:,.0f} ({projected / nlv:.0%} of NLV), "
                    f"exceeding the {RISK.max_total_exposure_pct:.0%} limit. "
                    f"Remaining headroom: ${headroom:,.0f}."
                ),
            }

    return None  # All checks passed


def _submit_and_wait(
    ib: IB, contract, order, wait_sec: float = 3.0
) -> dict[str, Any]:
    """Submit an order and wait briefly for initial status."""
    trade = ib.placeOrder(contract, order)
    ib.sleep(wait_sec)
    return {
        "order_id": trade.order.orderId,
        "perm_id": trade.order.permId,
        "order_status": trade.orderStatus.status,
        "filled_qty": float(trade.orderStatus.filled),
        "avg_fill_price": float(trade.orderStatus.avgFillPrice),
    }


# ---------------------------------------------------------------------------
# Tool: place_trade (market or limit)
# ---------------------------------------------------------------------------

@tool
@ensure_connected
def place_trade(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "MKT",
    limit_price: float | None = None,
    reason: str = "",
) -> dict:
    """
    Place a paper trade after enforcing hard risk limits.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        side: "BUY" or "SELL".
        quantity: Number of shares (must be a positive integer — IBKR
                  does not support fractional shares for equities).
        order_type: "MKT" for market orders, "LMT" for limit orders.
        limit_price: Required when order_type is "LMT".
        reason: Your trade thesis — logged to the audit trail.

    Returns:
        Dict with status ("SUBMITTED" or "REJECTED"), order details,
        and the thesis for audit purposes.
    """
    side = side.upper().strip()
    order_type = order_type.upper().strip()

    # --- Input validation ---
    if side not in {"BUY", "SELL"}:
        return {"status": "REJECTED", "reason": f"Invalid side: '{side}'. Must be 'BUY' or 'SELL'."}
    if quantity <= 0 or not isinstance(quantity, int):
        return {"status": "REJECTED", "reason": f"Quantity must be a positive integer, got: {quantity}"}
    if order_type == "LMT" and limit_price is None:
        return {"status": "REJECTED", "reason": "limit_price is required for LMT orders."}
    if order_type not in {"MKT", "LMT"}:
        return {"status": "REJECTED", "reason": f"Unsupported order_type: '{order_type}'. Use 'MKT' or 'LMT'."}

    # --- Contract resolution ---
    contract = qualify_us_equity(symbol)
    if contract is None:
        return {"status": "REJECTED", "reason": f"Cannot resolve IBKR contract for '{symbol}'."}

    if contract.secType not in RISK.allowed_sec_types:
        return {"status": "REJECTED", "reason": f"Security type '{contract.secType}' is not in the allowed set."}

    # --- Price for risk calculation ---
    price = get_market_price(contract)
    if price is None:
        return {"status": "REJECTED", "reason": f"Cannot determine market price for {symbol}. Data may be unavailable."}

    nlv = get_nlv()
    estimated_price = limit_price if order_type == "LMT" else price
    notional = estimated_price * quantity

    # --- Risk validation ---
    rejection = _validate_trade(symbol, side, quantity, estimated_price, nlv)
    if rejection is not None:
        log_trade("order_rejected", {**rejection, "symbol": symbol, "side": side, "quantity": quantity, "thesis": reason})
        return rejection

    # --- Build and submit order ---
    ib = get_connection()
    if order_type == "LMT":
        order = LimitOrder(side, quantity, limit_price)
    else:
        order = MarketOrder(side, quantity)

    order.tif = "DAY"  # Day orders only — no GTC for automated agents

    fill_info = _submit_and_wait(ib, contract, order)

    result = {
        "status": "SUBMITTED",
        "symbol": symbol.upper(),
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "estimated_notional": round(notional, 2),
        "estimated_price": round(estimated_price, 2),
        "position_pct_of_nlv": round(notional / nlv * 100, 2),
        "thesis": reason,
        **fill_info,
    }

    log_trade("order_submitted", result)
    logger.info(
        "Order submitted: %s %d %s @ %s ($%.0f, %.1f%% of NLV) — %s",
        side, quantity, symbol, order_type, notional,
        notional / nlv * 100, reason[:80],
    )

    return result


# ---------------------------------------------------------------------------
# Tool: place_bracket_trade (entry + take-profit + stop-loss)
# ---------------------------------------------------------------------------

@tool
@ensure_connected
def place_bracket_trade(
    symbol: str,
    side: str,
    quantity: int,
    entry_limit_price: float,
    take_profit_price: float,
    stop_loss_price: float,
    reason: str = "",
) -> dict:
    """
    Place an atomic bracket order: entry (limit) + take-profit (limit) + stop-loss (stop).

    All three legs are linked — if the entry fills, both exit orders activate.
    If entry is cancelled, the exits are cancelled automatically. This is the
    preferred order type for disciplined position management.

    Args:
        symbol: Ticker symbol.
        side: "BUY" or "SELL" (for the entry leg).
        quantity: Number of shares (positive integer).
        entry_limit_price: Limit price for the entry order.
        take_profit_price: Limit price for the take-profit exit.
        stop_loss_price: Stop price for the stop-loss exit.
        reason: Trade thesis for the audit log.

    Returns:
        Dict with order IDs for all three legs, or a rejection.
    """
    side = side.upper().strip()
    exit_side = "SELL" if side == "BUY" else "BUY"

    # --- Validation ---
    if side not in {"BUY", "SELL"}:
        return {"status": "REJECTED", "reason": f"Invalid side: '{side}'."}
    if quantity <= 0 or not isinstance(quantity, int):
        return {"status": "REJECTED", "reason": f"Quantity must be a positive integer."}

    # Price logic sanity
    if side == "BUY":
        if take_profit_price <= entry_limit_price:
            return {"status": "REJECTED", "reason": "Take-profit must be above entry price for a BUY bracket."}
        if stop_loss_price >= entry_limit_price:
            return {"status": "REJECTED", "reason": "Stop-loss must be below entry price for a BUY bracket."}
    else:
        if take_profit_price >= entry_limit_price:
            return {"status": "REJECTED", "reason": "Take-profit must be below entry price for a SELL bracket."}
        if stop_loss_price <= entry_limit_price:
            return {"status": "REJECTED", "reason": "Stop-loss must be above entry price for a SELL bracket."}

    # --- Contract ---
    contract = qualify_us_equity(symbol)
    if contract is None:
        return {"status": "REJECTED", "reason": f"Cannot resolve contract for '{symbol}'."}

    nlv = get_nlv()
    notional = entry_limit_price * quantity

    # --- Risk ---
    rejection = _validate_trade(symbol, side, quantity, entry_limit_price, nlv)
    if rejection is not None:
        log_trade("bracket_rejected", {**rejection, "symbol": symbol, "thesis": reason})
        return rejection

    # --- Build bracket orders ---
    ib = get_connection()

    # Parent: entry order
    parent = LimitOrder(side, quantity, entry_limit_price)
    parent.orderId = ib.client.getReqId()
    parent.tif = "DAY"
    parent.transmit = False  # Don't send until all legs are ready

    # Take-profit (child 1)
    tp = LimitOrder(exit_side, quantity, take_profit_price)
    tp.orderId = ib.client.getReqId()
    tp.parentId = parent.orderId
    tp.tif = "GTC"  # Exits should persist
    tp.transmit = False

    # Stop-loss (child 2) — transmit=True on last leg sends all three
    sl = StopOrder(exit_side, quantity, stop_loss_price)
    sl.orderId = ib.client.getReqId()
    sl.parentId = parent.orderId
    sl.tif = "GTC"
    sl.transmit = True  # This triggers transmission of the entire bracket

    # Submit all legs
    parent_trade = ib.placeOrder(contract, parent)
    tp_trade = ib.placeOrder(contract, tp)
    sl_trade = ib.placeOrder(contract, sl)
    ib.sleep(3)

    result = {
        "status": "SUBMITTED",
        "symbol": symbol.upper(),
        "side": side,
        "quantity": quantity,
        "entry_limit_price": entry_limit_price,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "estimated_notional": round(notional, 2),
        "position_pct_of_nlv": round(notional / nlv * 100, 2),
        "risk_reward_ratio": round(
            abs(take_profit_price - entry_limit_price) /
            abs(entry_limit_price - stop_loss_price), 2
        ) if abs(entry_limit_price - stop_loss_price) > 0 else "infinite",
        "max_loss": round(abs(entry_limit_price - stop_loss_price) * quantity, 2),
        "max_gain": round(abs(take_profit_price - entry_limit_price) * quantity, 2),
        "orders": {
            "entry": {"order_id": parent_trade.order.orderId, "status": parent_trade.orderStatus.status},
            "take_profit": {"order_id": tp_trade.order.orderId, "status": tp_trade.orderStatus.status},
            "stop_loss": {"order_id": sl_trade.order.orderId, "status": sl_trade.orderStatus.status},
        },
        "thesis": reason,
    }

    log_trade("bracket_submitted", result)
    logger.info(
        "Bracket submitted: %s %d %s | entry=%.2f tp=%.2f sl=%.2f | R:R=%.1fx — %s",
        side, quantity, symbol, entry_limit_price, take_profit_price,
        stop_loss_price, result["risk_reward_ratio"] if isinstance(result["risk_reward_ratio"], float) else 0,
        reason[:60],
    )

    return result


# ---------------------------------------------------------------------------
# Tool: close_position
# ---------------------------------------------------------------------------

@tool
@ensure_connected
def close_position(symbol: str, reason: str = "") -> dict:
    """
    Fully close an existing position in a symbol with a market order.

    Use this to exit a position entirely. For partial exits, use place_trade
    with the appropriate quantity.

    Args:
        symbol: Ticker symbol to close.
        reason: Why the position is being closed (for audit trail).

    Returns:
        Dict with the closing order details, or an error if no position exists.
    """
    pos = get_position_for_symbol(symbol)
    if pos is None:
        return {"status": "NO_POSITION", "reason": f"No open position found for {symbol}."}

    qty = pos["qty"]
    contract = pos["contract"]

    # Close direction: sell if long, buy if short
    close_side = "SELL" if qty > 0 else "BUY"
    close_qty = abs(int(qty))

    ib = get_connection()
    order = MarketOrder(close_side, close_qty)
    order.tif = "DAY"

    fill_info = _submit_and_wait(ib, contract, order)

    result = {
        "status": "SUBMITTED",
        "action": "CLOSE_POSITION",
        "symbol": symbol.upper(),
        "side": close_side,
        "quantity": close_qty,
        "position_was": "LONG" if qty > 0 else "SHORT",
        "avg_entry": pos["avg_cost"],
        "exit_price_est": pos["market_price"],
        "unrealized_pl_at_close": pos["unrealized_pl"],
        "thesis": reason,
        **fill_info,
    }

    log_trade("position_closed", result)
    logger.info(
        "Position closed: %s %d %s (was %s, P&L: %s) — %s",
        close_side, close_qty, symbol,
        "LONG" if qty > 0 else "SHORT",
        pos["unrealized_pl"], reason[:80],
    )

    return result
