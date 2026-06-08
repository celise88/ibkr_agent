"""
tools/leveraged_etf_flow.py

Leveraged ETF end-of-day rebalancing flow estimator.

For each leveraged ETF in the watchlist, computes:
  - Intraday return of the underlying
  - Estimated rebalancing notional (NAV × L × (L-1) × R)
  - Direction of end-of-day flow (buy/sell pressure on the underlying)
  - Volatility decay drag estimate
  - Actionable trade signal context

The rebalancing flow formula:
    flow = NAV × leverage × (leverage - 1) × intraday_return_underlying

Positive flow → ETF must BUY underlying near close (bullish pressure)
Negative flow → ETF must SELL underlying near close (bearish pressure)

This is a deterministic, math-driven tool — no LLM inference here.
The analyst agent uses this output as structured context.

Usage (as LangChain tool):
    from ibkr_agent.tools.leveraged_etf_flow import get_etf_flow_analysis

Usage (standalone):
    python -m ibkr_agent.tools.leveraged_etf_flow
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from ib_insync import IB, Stock, util

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watchlist configuration
# ---------------------------------------------------------------------------
# Each entry: (etf_symbol, underlying_symbol, leverage_multiple, approx_aum_millions)
# AUM is used when live NAV/total-net-assets is unavailable from IBKR.
# Update AUM periodically — Direxion publishes daily on their website.
#
# Format: (etf_ticker, underlying_ticker, leverage, aum_usd_millions)

LEVERAGED_ETF_WATCHLIST: list[tuple[str, str, float, float]] = [
    # ── Single-stock 2x bull ETFs (NVDA) ──────────────────────────────────────
    ("NVDU",  "NVDA",  2.0,  547.0),   # Direxion 2x NVDA (~$547M AUM)
    ("NVDL",  "NVDA",  2.0,  5800.0),  # GraniteShares 2x NVDA (~$5.8B AUM)

    # ── Single-stock 2x bull ETFs (other mega-cap tech) ──────────────────────
    ("GGLL",  "GOOGL", 2.0,  1330.0),  # Direxion 2x GOOGL (~$1.33B AUM)
    ("AAPU",  "AAPL",  2.0,  160.0),   # Direxion 2x AAPL (~$160M AUM)
    ("AMZU",  "AMZN",  2.0,  300.0),   # Direxion 2x AMZN (~$300M AUM)
    ("METU",  "META",  2.0,  365.0),   # Direxion 2x META (~$365M AUM)
    ("MSFU",  "MSFT",  2.0,  180.0),   # GraniteShares 2x MSFT (~$180M AUM)
    ("AVL",   "AVGO",  2.0,  140.0),   # Direxion 2x AVGO (~$140M AUM)
    ("TSMX",  "TSM",   2.0,  442.0),   # Direxion 2x TSM (~$442M AUM)

    # ── Single-stock 2x bull ETFs (other) ─────────────────────────────────────
    ("TSLL",  "TSLA",  2.0,  3900.0),  # Direxion 2x TSLA (~$3.9B AUM)
    ("PLTU",  "PLTR",  2.0,  400.0),   # Direxion 2x PLTR (~$400M AUM)
    ("MSTU",  "MSTR",  2.0,  660.0),   # T-Rex 2x MSTR (~$660M AUM)

    # ── Sector/index leveraged ETFs ───────────────────────────────────────────
    ("SOXL",  "SOXX",  3.0,  7200.0),  # Direxion 3x Semiconductors (~$7.2B AUM)
    ("TECL",  "XLK",   3.0,  6800.0),  # Direxion 3x Technology Select Sector (~$6.8B AUM)
    ("TQQQ",  "QQQ",   3.0,  22000.0), # ProShares 3x Nasdaq (~$22B AUM)
    ("UPRO",  "SPY",   3.0,  3800.0),  # ProShares 3x S&P 500 (~$3.8B AUM)
    ("LABU",  "XBI",   3.0,  700.0),   # Direxion 3x Biotech (~$700M AUM)
    ("TNA",   "IWM",   3.0,  1100.0),  # Direxion 3x Russell 2000 (~$1.1B AUM)
    ("FNGU",  "FANG+", 3.0,  2700.0),  # MicroSectors 3x FANG+ (~$2.7B AUM)
]

# Minimum absolute flow (USD millions) to flag as a meaningful signal
FLOW_SIGNAL_THRESHOLD_M: float = 10.0

# Minimum intraday move (%) to consider significant enough for flow context
MIN_MOVE_PCT: float = 1.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ETFFlowEstimate:
    etf_symbol: str
    underlying_symbol: str
    leverage: float
    aum_millions: float

    # Prices
    etf_price: float | None = None
    etf_change_pct: float | None = None
    underlying_price: float | None = None
    underlying_change_pct: float | None = None

    # Flow
    estimated_flow_millions: float | None = None  # + = buy pressure, - = sell pressure
    flow_direction: str = "UNKNOWN"               # BUY | SELL | NEUTRAL | UNKNOWN

    # Vol decay
    underlying_annual_iv: float | None = None
    underlying_hist_vol_annual: float | None = None
    daily_decay_pct: float | None = None           # volatility drag per day

    # Signal
    signal_strength: str = "NONE"                 # STRONG | MODERATE | WEAK | NONE
    signal_context: str = ""

    # Metadata
    fetch_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ETFFlowReport:
    generated_at: str
    market_session: str   # PRE | OPEN | CLOSE_APPROACH | AFTER
    etf_count: int
    strong_signals: list[dict]
    moderate_signals: list[dict]
    all_estimates: list[dict]
    briefing_text: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def estimate_rebalancing_flow(
    nav_millions: float,
    leverage: float,
    underlying_return: float,
) -> float:
    """
    Estimate end-of-day rebalancing flow in USD millions.

    Derivation:
        After a return R, the ETF's NAV grows to NAV*(1 + L*R).
        Its derivative exposure grew to NAV*L*(1 + R).
        To restore L× leverage on the new NAV:
            Target exposure = NAV*(1+L*R) * L
            Current exposure = NAV*L*(1+R)
            Delta = NAV*L*(L*R - R) = NAV * L * (L-1) * R

    Args:
        nav_millions: Approximate AUM in $M (used as NAV proxy)
        leverage: The stated daily leverage multiple (e.g. 2.0, 3.0)
        underlying_return: Fractional intraday return of the underlying (e.g. -0.056)

    Returns:
        Flow in $M. Positive = must buy, Negative = must sell.
    """
    return nav_millions * leverage * (leverage - 1.0) * underlying_return


def estimate_daily_vol_decay(leverage: float, annual_vol: float) -> float:
    """
    Approximate daily volatility drag as a percentage.

    Formula: ½ × (L² - L) × σ_daily²
    where σ_daily = annual_vol / sqrt(252)

    Returns decay as a positive percentage (it's always a cost).
    """
    sigma_daily = annual_vol / math.sqrt(252)
    return 0.5 * (leverage ** 2 - leverage) * (sigma_daily ** 2) * 100.0


def classify_signal_strength(
    flow_millions: float,
    underlying_change_pct: float,
    aum_millions: float,
) -> str:
    """
    Classify signal strength based on flow magnitude and move size.

    Strong:   |flow| > 5% of AUM AND |underlying move| > 3%
    Moderate: |flow| > THRESHOLD_M AND |underlying move| > MIN_MOVE_PCT
    Weak:     |flow| > THRESHOLD_M OR |underlying move| > MIN_MOVE_PCT
    None:     below all thresholds
    """
    abs_flow = abs(flow_millions)
    abs_move = abs(underlying_change_pct)

    flow_pct_of_aum = (abs_flow / aum_millions * 100.0) if aum_millions else 0.0

    if flow_pct_of_aum > 5.0 and abs_move > 3.0:
        return "STRONG"
    if abs_flow > FLOW_SIGNAL_THRESHOLD_M and abs_move > MIN_MOVE_PCT:
        return "MODERATE"
    if abs_flow > FLOW_SIGNAL_THRESHOLD_M or abs_move > MIN_MOVE_PCT:
        return "WEAK"
    return "NONE"


def build_signal_context(estimate: ETFFlowEstimate) -> str:
    """
    Generate a human-readable, actionable context string for the signal.
    This is what the analyst agent reads directly.
    """
    if estimate.error or estimate.estimated_flow_millions is None:
        return f"Data unavailable: {estimate.error or 'unknown error'}"

    parts = []
    flow = estimate.estimated_flow_millions
    move = estimate.underlying_change_pct or 0.0
    decay = estimate.daily_decay_pct

    direction_verb = "BUY" if flow > 0 else "SELL"
    abs_flow = abs(flow)

    parts.append(
        f"{estimate.etf_symbol} ({estimate.leverage:.0f}x {estimate.underlying_symbol}): "
        f"underlying {'+' if move >= 0 else ''}{move:.2f}% → "
        f"estimated ~${abs_flow:.1f}M {direction_verb} pressure on {estimate.underlying_symbol} near close."
    )

    if estimate.signal_strength in ("STRONG", "MODERATE"):
        if flow < 0:
            parts.append(
                f"  ⚠ Mechanical SELLING into close — avoid chasing {estimate.underlying_symbol} "
                f"long in final 30-60 min. Better entry likely post-close or next morning."
            )
        else:
            parts.append(
                f"  ↑ Mechanical BUYING into close — {estimate.underlying_symbol} may show "
                f"artificial late-day strength. Front-runners likely already positioned."
            )

    if decay is not None and estimate.leverage > 1.0:
        parts.append(
            f"  Vol decay: ~{decay:.3f}%/day (~{decay*252:.1f}%/yr drag at current vol). "
            f"Hold {estimate.etf_symbol} for day-trades only."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# IBKR data fetcher
# ---------------------------------------------------------------------------

def _fetch_snapshot(ib: IB, symbol: str, exchange: str = "SMART") -> dict[str, Any]:
    """
    Fetch a market data snapshot for a symbol via ib_insync.
    Returns dict with: price, change_pct, annual_iv, hist_vol_annual.

    Generic tick list:
        104 = Historical Volatility (30-day annualized)
        106 = Option Implied Volatility
        165 = Misc Stats
        293 = Trade Count
        294 = Trade Rate
        295 = Last RTH Trade
    """
    contract = Stock(symbol, exchange, "USD")
    ib.qualifyContracts(contract)

    ticker = ib.reqMktData(contract, genericTickList="104,106,165,293,294,295", snapshot=True)
    ib.sleep(2.0)  # Allow snapshot to populate

    result: dict[str, Any] = {
        "symbol": symbol,
        "price": None,
        "change_pct": None,
        "annual_iv": None,
        "hist_vol_annual": None,
    }

    price = ticker.marketPrice()
    if price and not math.isnan(price):
        result["price"] = round(price, 4)

    close = ticker.close
    if close and not math.isnan(close) and result["price"]:
        result["change_pct"] = round((result["price"] - close) / close * 100.0, 4)

    if ticker.impliedVolatility and not math.isnan(ticker.impliedVolatility):
        result["annual_iv"] = round(ticker.impliedVolatility, 6)

    if ticker.histVolatility and not math.isnan(ticker.histVolatility):
        result["hist_vol_annual"] = round(ticker.histVolatility, 6)

    ib.cancelMktData(contract)
    return result


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def compute_etf_flow_estimates(
    ib: IB,
    watchlist: list[tuple[str, str, float, float]] | None = None,
) -> list[ETFFlowEstimate]:
    """
    For each ETF in the watchlist, fetch live data and compute flow estimates.

    Args:
        ib: Connected ib_insync IB instance
        watchlist: List of (etf_symbol, underlying_symbol, leverage, aum_millions).
                   Defaults to module-level LEVERAGED_ETF_WATCHLIST.

    Returns:
        List of ETFFlowEstimate dataclasses, sorted by |flow| descending.
    """
    if watchlist is None:
        watchlist = LEVERAGED_ETF_WATCHLIST

    # Filter out any non-leveraged placeholders (leverage <= 1.0)
    watchlist = [(e, u, l, a) for e, u, l, a in watchlist if l > 1.0]

    estimates: list[ETFFlowEstimate] = []

    # Cache underlying fetches — multiple ETFs may share an underlying (e.g. SOXX/QQQ)
    underlying_cache: dict[str, dict] = {}

    for etf_sym, underlying_sym, leverage, aum_m in watchlist:
        estimate = ETFFlowEstimate(
            etf_symbol=etf_sym,
            underlying_symbol=underlying_sym,
            leverage=leverage,
            aum_millions=aum_m or 100.0,
        )

        try:
            etf_data = _fetch_snapshot(ib, etf_sym)
            estimate.etf_price = etf_data["price"]
            estimate.etf_change_pct = etf_data["change_pct"]

            if underlying_sym not in underlying_cache:
                underlying_cache[underlying_sym] = _fetch_snapshot(ib, underlying_sym)
            underlying_data = underlying_cache[underlying_sym]

            estimate.underlying_price = underlying_data["price"]
            estimate.underlying_change_pct = underlying_data["change_pct"]
            estimate.underlying_annual_iv = underlying_data["annual_iv"]
            estimate.underlying_hist_vol_annual = underlying_data["hist_vol_annual"]

            # Use the higher of IV or hist vol for decay calculation (conservative)
            vol_for_decay = max(
                filter(None, [
                    underlying_data["annual_iv"],
                    underlying_data["hist_vol_annual"],
                    0.30,  # floor at 30% annual vol — single-stock minimum assumption
                ])
            )

            if estimate.underlying_change_pct is not None:
                underlying_return = estimate.underlying_change_pct / 100.0
                estimate.estimated_flow_millions = round(
                    estimate_rebalancing_flow(aum_m or 100.0, leverage, underlying_return), 2
                )
                estimate.flow_direction = (
                    "BUY" if estimate.estimated_flow_millions > 0
                    else "SELL" if estimate.estimated_flow_millions < 0
                    else "NEUTRAL"
                )
                estimate.signal_strength = classify_signal_strength(
                    estimate.estimated_flow_millions,
                    estimate.underlying_change_pct,
                    aum_m or 100.0,
                )

            estimate.daily_decay_pct = round(
                estimate_daily_vol_decay(leverage, vol_for_decay), 4
            )

            estimate.signal_context = build_signal_context(estimate)

        except Exception as exc:
            logger.error("Failed to process %s/%s: %s", etf_sym, underlying_sym, exc, exc_info=True)
            estimate.error = str(exc)
            estimate.signal_context = f"Error: {exc}"

        estimates.append(estimate)

    estimates.sort(
        key=lambda e: abs(e.estimated_flow_millions or 0.0),
        reverse=True,
    )

    return estimates


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _infer_market_session() -> str:
    """Classify the current time relative to NYSE session (US Eastern)."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    total_minutes = now_et.hour * 60 + now_et.minute

    if total_minutes < 9 * 60 + 30:
        return "PRE"
    if total_minutes < 15 * 60:
        return "OPEN"
    if total_minutes < 15 * 60 + 30:
        return "CLOSE_APPROACH"
    return "AFTER"


def build_flow_report(estimates: list[ETFFlowEstimate]) -> ETFFlowReport:
    """
    Aggregate individual estimates into a structured briefing report.
    """
    session = _infer_market_session()

    strong = [e.to_dict() for e in estimates if e.signal_strength == "STRONG"]
    moderate = [e.to_dict() for e in estimates if e.signal_strength == "MODERATE"]

    lines = [
        "=== LEVERAGED ETF REBALANCING FLOW BRIEFING ===",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Market session: {session}",
        f"ETFs analyzed: {len(estimates)}",
        "",
    ]

    if not strong and not moderate:
        lines.append(
            "No significant rebalancing flows detected today "
            f"(threshold: >${FLOW_SIGNAL_THRESHOLD_M:.0f}M flow or >{MIN_MOVE_PCT:.1f}% underlying move). "
            "Market likely quiet — normal session dynamics apply."
        )
    else:
        if strong:
            lines.append("⚡ STRONG SIGNALS (>5% of AUM flow AND >3% underlying move):")
            for e in estimates:
                if e.signal_strength == "STRONG":
                    lines.append(f"  • {e.signal_context}")
            lines.append("")

        if moderate:
            lines.append("⚠ MODERATE SIGNALS:")
            for e in estimates:
                if e.signal_strength == "MODERATE":
                    lines.append(f"  • {e.signal_context}")
            lines.append("")

    if session == "CLOSE_APPROACH":
        sell_etfs = [e for e in estimates if e.flow_direction == "SELL" and e.signal_strength in ("STRONG", "MODERATE")]
        buy_etfs  = [e for e in estimates if e.flow_direction == "BUY"  and e.signal_strength in ("STRONG", "MODERATE")]
        if sell_etfs:
            underlyings = list({e.underlying_symbol for e in sell_etfs})
            lines.append(
                f"⛔ CLOSE WARNING: Mechanical SELLING pressure on {', '.join(underlyings)} "
                f"expected in next 30-45 min from ETF rebalancing. "
                f"Avoid initiating long positions in these names into the close."
            )
        if buy_etfs:
            underlyings = list({e.underlying_symbol for e in buy_etfs})
            lines.append(
                f"📈 CLOSE NOTE: Mechanical BUYING pressure on {', '.join(underlyings)} "
                f"expected near close. Potential front-run opportunity has likely passed; "
                f"watch for post-close reversal."
            )
    elif session == "OPEN":
        lines.append(
            "SESSION CONTEXT: Early session — rebalancing flows will build as intraday moves develop. "
            "Re-run this analysis at 2:30-3:00 PM ET for meaningful close estimates."
        )

    decay_warnings = [
        e for e in estimates
        if e.daily_decay_pct is not None and e.daily_decay_pct > 0.5
    ]
    if decay_warnings:
        lines.append("")
        lines.append("📉 HIGH DECAY WARNINGS (avoid multi-day holds):")
        for e in decay_warnings:
            lines.append(
                f"  • {e.etf_symbol}: ~{e.daily_decay_pct:.3f}%/day "
                f"(~{e.daily_decay_pct * 252:.1f}%/yr) at current vol — "
                f"intraday use only."
            )

    return ETFFlowReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        market_session=session,
        etf_count=len(estimates),
        strong_signals=strong,
        moderate_signals=moderate,
        all_estimates=[e.to_dict() for e in estimates],
        briefing_text="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------

from langchain_core.tools import tool as lc_tool


@lc_tool
def get_etf_flow_analysis(
    watchlist_override: str | None = None,
) -> str:
    """
    Compute leveraged ETF rebalancing flow estimates for the configured watchlist.

    Returns a structured briefing of which ETFs have meaningful end-of-day
    rebalancing flows — mechanical buy or sell pressure on their underlying stocks.
    Use this to inform intraday entry/exit timing and avoid trading against
    large mechanical flows near market close.

    Args:
        watchlist_override: Optional JSON string of additional ETFs to include,
            format: [["ETF_SYM", "UNDERLYING_SYM", leverage_float, aum_millions_float], ...]
            These are ADDED to the default watchlist, not replacing it.

    Returns:
        Formatted briefing text with flow estimates and actionable context.
    """
    from ibkr_agent.connection import get_connection

    ib = get_connection()

    watchlist = list(LEVERAGED_ETF_WATCHLIST)

    if watchlist_override:
        try:
            extra = json.loads(watchlist_override)
            watchlist.extend(
                (row[0], row[1], float(row[2]), float(row[3]) if row[3] else 100.0)
                for row in extra
            )
            logger.info("Added %d ETFs from watchlist_override", len(extra))
        except (json.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Invalid watchlist_override, ignoring: %s", exc)

    estimates = compute_etf_flow_estimates(ib, watchlist)
    report = build_flow_report(estimates)

    return report.briefing_text


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from ibkr_agent.audit import setup_logging
    from ibkr_agent.connection import get_connection, disconnect

    setup_logging()

    print("Connecting to IBKR...")
    ib = get_connection()

    try:
        estimates = compute_etf_flow_estimates(ib)
        report = build_flow_report(estimates)
        print(report.briefing_text)
        print()
        print("--- JSON output ---")
        print(report.to_json())
    finally:
        disconnect()
        sys.exit(0)
