"""
Earnings and event catalyst tool.

Surfaces upcoming and recent catalysts that create informational asymmetries
the LLM can exploit through synthesis:

  - Earnings dates and whether the company reports before/after market
  - Analyst consensus estimates (EPS, revenue) to compare against actuals
  - Recent earnings surprises — the market's reaction to beats/misses
  - Upcoming catalysts: FDA dates, product launches, conferences

Data sources:
  - Finnhub free tier (earnings calendar, estimates, recommendations)
  - Alpha Vantage free tier (earnings, overview)
  - SEC EDGAR 8-K filings (actual results)

The LLM's job: read the NUMBERS and form a view on whether the market
has correctly priced the information. A 10% earnings beat means nothing
if guidance was cut. A small miss with raised guidance is bullish.

Requires: FINNHUB_API_KEY and/or ALPHAVANTAGE_API_KEY in environment.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from langchain_core.tools import tool

from ibkr_agent.audit import log_agent

logger = logging.getLogger(__name__)

FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")


# ---------------------------------------------------------------------------
# Finnhub API helpers
# ---------------------------------------------------------------------------

def _finnhub_get(endpoint: str, params: dict[str, str]) -> dict | list | None:
    """Generic Finnhub API GET with error handling."""
    if not FINNHUB_KEY:
        return None
    params["token"] = FINNHUB_KEY
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://finnhub.io/api/v1/{endpoint}?{query}"
    try:
        req = Request(url, headers={"User-Agent": "IBKRAgent/0.1"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("Finnhub %s failed: %s", endpoint, exc)
        return None


def _fetch_earnings_calendar(
    from_date: str, to_date: str
) -> list[dict[str, Any]]:
    """Fetch earnings calendar from Finnhub."""
    data = _finnhub_get("calendar/earnings", {
        "from": from_date,
        "to": to_date,
    })
    if not data or "earningsCalendar" not in data:
        return []
    return data["earningsCalendar"]


def _fetch_earnings_surprises(symbol: str, limit: int = 4) -> list[dict[str, Any]]:
    """Fetch recent earnings surprises (actual vs estimate)."""
    data = _finnhub_get("stock/earnings", {"symbol": symbol, "limit": str(limit)})
    if not data or not isinstance(data, list):
        return []
    return data


def _fetch_analyst_recommendations(symbol: str) -> list[dict[str, Any]]:
    """Fetch analyst recommendation trends."""
    data = _finnhub_get("stock/recommendation", {"symbol": symbol})
    if not data or not isinstance(data, list):
        return []
    return data[:6]  # Last 6 months


def _fetch_price_target(symbol: str) -> dict[str, Any] | None:
    """Fetch analyst price target consensus."""
    data = _finnhub_get("stock/price-target", {"symbol": symbol})
    if not data or not isinstance(data, dict):
        return None
    return data


def _fetch_company_profile(symbol: str) -> dict[str, Any] | None:
    """Fetch basic company profile."""
    data = _finnhub_get("stock/profile2", {"symbol": symbol})
    if not data or not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Alpha Vantage helpers
# ---------------------------------------------------------------------------

def _alphavantage_get(function: str, params: dict[str, str]) -> dict | None:
    """Generic Alpha Vantage GET."""
    if not ALPHAVANTAGE_KEY:
        return None
    params["function"] = function
    params["apikey"] = ALPHAVANTAGE_KEY
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://www.alphavantage.co/query?{query}"
    try:
        req = Request(url, headers={"User-Agent": "IBKRAgent/0.1"})
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("Alpha Vantage %s failed: %s", function, exc)
        return None


def _fetch_company_overview(symbol: str) -> dict[str, Any] | None:
    """Fetch comprehensive company overview from Alpha Vantage."""
    data = _alphavantage_get("OVERVIEW", {"symbol": symbol})
    if not data or "Symbol" not in data:
        return None

    # Extract the fields most useful for LLM synthesis
    return {
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": data.get("MarketCapitalization"),
        "pe_ratio": data.get("PERatio"),
        "forward_pe": data.get("ForwardPE"),
        "peg_ratio": data.get("PEGRatio"),
        "eps": data.get("EPS"),
        "revenue_ttm": data.get("RevenueTTM"),
        "profit_margin": data.get("ProfitMargin"),
        "operating_margin": data.get("OperatingMarginTTM"),
        "roe": data.get("ReturnOnEquityTTM"),
        "dividend_yield": data.get("DividendYield"),
        "beta": data.get("Beta"),
        "52wk_high": data.get("52WeekHigh"),
        "52wk_low": data.get("52WeekLow"),
        "50d_ma": data.get("50DayMovingAverage"),
        "200d_ma": data.get("200DayMovingAverage"),
        "shares_outstanding": data.get("SharesOutstanding"),
        "analyst_target": data.get("AnalystTargetPrice"),
        "analyst_rating": data.get("AnalystRatingStrongBuy"),
    }


# ---------------------------------------------------------------------------
# Tool: earnings and catalyst analysis
# ---------------------------------------------------------------------------

@tool
def get_earnings_analysis(symbol: str) -> dict:
    """
    Comprehensive earnings and analyst data for a symbol.

    Returns recent earnings history (actual vs estimates), analyst
    recommendations, price targets, and company fundamentals. This is
    where the LLM adds the most value — by reading the PATTERN in
    earnings surprises, not just the latest number.

    Key synthesis tasks for the LLM:
    - Is the company consistently beating or missing estimates?
    - Are beats accelerating or decelerating?
    - Do analyst recommendations lag the actual earnings trajectory?
    - Is the price target consensus above or below current price?
    - What does the forward PE vs trailing PE imply about growth expectations?

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
    """
    symbol = symbol.upper().strip()

    # Earnings surprises (actual vs estimate)
    surprises = _fetch_earnings_surprises(symbol, limit=8)
    formatted_surprises = []
    for s in surprises:
        actual = s.get("actual")
        estimate = s.get("estimate")
        surprise_pct = s.get("surprisePercent")
        formatted_surprises.append({
            "period": s.get("period", "unknown"),
            "actual_eps": actual,
            "estimated_eps": estimate,
            "surprise_pct": surprise_pct,
            "beat_or_miss": (
                "BEAT" if surprise_pct and surprise_pct > 0
                else "MISS" if surprise_pct and surprise_pct < 0
                else "IN_LINE"
            ),
        })

    # Analyst recommendations trend
    recommendations = _fetch_analyst_recommendations(symbol)
    rec_summary = []
    for r in recommendations:
        rec_summary.append({
            "period": r.get("period", "unknown"),
            "strong_buy": r.get("strongBuy", 0),
            "buy": r.get("buy", 0),
            "hold": r.get("hold", 0),
            "sell": r.get("sell", 0),
            "strong_sell": r.get("strongSell", 0),
        })

    # Price target
    price_target = _fetch_price_target(symbol)
    target_data = None
    if price_target:
        target_data = {
            "target_high": price_target.get("targetHigh"),
            "target_low": price_target.get("targetLow"),
            "target_mean": price_target.get("targetMean"),
            "target_median": price_target.get("targetMedian"),
            "last_updated": price_target.get("lastUpdated"),
        }

    # Company profile
    profile = _fetch_company_profile(symbol)
    overview = _fetch_company_overview(symbol)

    # Earnings trend analysis
    beat_count = sum(1 for s in formatted_surprises if s["beat_or_miss"] == "BEAT")
    miss_count = sum(1 for s in formatted_surprises if s["beat_or_miss"] == "MISS")
    total = len(formatted_surprises)

    data_sources = []
    if FINNHUB_KEY:
        data_sources.append("finnhub")
    if ALPHAVANTAGE_KEY:
        data_sources.append("alphavantage")

    result = {
        "symbol": symbol,
        "data_sources": data_sources,
        "company_profile": profile,
        "company_overview": overview,
        "earnings_history": {
            "quarters_analyzed": total,
            "beats": beat_count,
            "misses": miss_count,
            "beat_rate": f"{beat_count/total*100:.0f}%" if total > 0 else "no_data",
            "details": formatted_surprises,
        },
        "analyst_recommendations": rec_summary,
        "price_targets": target_data,
        "synthesis_guidance": (
            "YOUR EDGE: Read the TRAJECTORY, not just the latest number. "
            "A company beating estimates by shrinking margins is a red flag. "
            "A company missing estimates but raising full-year guidance is "
            "potentially bullish. Compare the earnings surprise trend against "
            "analyst recommendation changes — if analysts are still upgrading "
            "after 4 consecutive beats, the market may not have fully priced "
            "in the momentum. If analysts are downgrading despite beats, "
            "they're seeing something in the guidance or margins you should "
            "investigate."
        ),
    }

    if not data_sources:
        result["warning"] = (
            "No API keys configured for earnings data. Set FINNHUB_API_KEY "
            "and/or ALPHAVANTAGE_API_KEY in your .env file. Both offer free "
            "tiers sufficient for this agent."
        )

    log_agent("earnings_analysis", {
        "symbol": symbol,
        "earnings_quarters": total,
        "beat_rate": f"{beat_count}/{total}",
    })

    return result


@tool
def get_earnings_calendar(days_ahead: int = 7) -> dict:
    """
    Fetch the earnings calendar for the next N days.

    Use this to identify upcoming catalysts. Companies reporting earnings
    in the next few days are high-information-value targets — pull their
    full earnings history and SEC filings before the report.

    Args:
        days_ahead: How many days ahead to scan (default 7, max 30).
    """
    days_ahead = min(max(days_ahead, 1), 30)
    today = datetime.now()
    from_date = today.strftime("%Y-%m-%d")
    to_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    calendar = _fetch_earnings_calendar(from_date, to_date)

    # Group by date and sort by market cap (if available)
    by_date: dict[str, list] = {}
    for entry in calendar:
        date = entry.get("date", "unknown")
        by_date.setdefault(date, []).append({
            "symbol": entry.get("symbol"),
            "hour": entry.get("hour", "unknown"),  # bmo=before, amc=after, dmh=during
            "eps_estimate": entry.get("epsEstimate"),
            "revenue_estimate": entry.get("revenueEstimate"),
            "eps_actual": entry.get("epsActual"),  # None if not yet reported
            "revenue_actual": entry.get("revenueActual"),
        })

    return {
        "period": f"{from_date} to {to_date}",
        "total_reports": len(calendar),
        "by_date": by_date,
        "strategy_note": (
            "Pre-earnings positioning is where information synthesis creates "
            "the most value. For each upcoming report: (1) Pull the earnings "
            "history to see beat/miss patterns, (2) Check SEC filings for "
            "recent 8-Ks or guidance updates, (3) Look at the technical "
            "setup — is the stock already pricing in a beat? (4) Consider "
            "whether the risk/reward favors a pre-earnings entry or waiting "
            "for the reaction."
        ),
    }
