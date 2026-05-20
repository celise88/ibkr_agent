"""
Macro environment and economic data tool.

Provides the broader economic context that affects all positions:
  - Key economic indicators from FRED (Federal Reserve Economic Data)
  - Recent and upcoming macro events (Fed meetings, CPI, jobs reports)
  - Yield curve, credit spreads, and volatility regime

The LLM's job: synthesize the macro backdrop into actionable context.
"Is this a risk-on or risk-off environment? Are rates rising into
earnings season? Is credit tightening while the equity market rallies?"
These cross-domain synthesis questions are where LLMs outperform
any single-indicator rule.

Requires: FRED_API_KEY in environment (free at https://fred.stlouisfed.org/docs/api/api_key.html)
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

FRED_KEY = os.environ.get("FRED_API_KEY", "")
_FRED_BASE = "https://api.stlouisfed.org/fred"


# ---------------------------------------------------------------------------
# FRED API helpers
# ---------------------------------------------------------------------------

def _fred_get_series(
    series_id: str,
    observation_start: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch recent observations for a FRED series."""
    if not FRED_KEY:
        return []

    if observation_start is None:
        observation_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(limit),
        "observation_start": observation_start,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_FRED_BASE}/series/observations?{query}"

    try:
        req = Request(url, headers={"User-Agent": "IBKRAgent/0.1"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("observations", [])
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("FRED series %s failed: %s", series_id, exc)
        return []


def _get_latest_value(series_id: str) -> dict[str, Any]:
    """Get the most recent non-null observation for a series."""
    obs = _fred_get_series(series_id, limit=5)
    for o in obs:
        val = o.get("value", ".")
        if val != ".":
            return {
                "value": float(val),
                "date": o.get("date"),
            }
    return {"value": None, "date": None}


def _get_series_with_change(series_id: str, periods: int = 3) -> dict[str, Any]:
    """Get latest value plus recent trend (change over N periods)."""
    obs = _fred_get_series(series_id, limit=periods + 1)
    valid = [o for o in obs if o.get("value", ".") != "."]

    if not valid:
        return {"latest": None, "date": None, "trend": "no_data"}

    latest_val = float(valid[0]["value"])
    result = {
        "latest": latest_val,
        "date": valid[0]["date"],
    }

    if len(valid) > 1:
        prev_val = float(valid[-1]["value"])
        result["prior"] = prev_val
        result["prior_date"] = valid[-1]["date"]
        result["change"] = round(latest_val - prev_val, 4)
        result["trend"] = "rising" if latest_val > prev_val else "falling" if latest_val < prev_val else "flat"

    return result


# ---------------------------------------------------------------------------
# Key macro series definitions
# ---------------------------------------------------------------------------

# These are the FRED series IDs for the indicators that matter most
# for equity market context
_MACRO_SERIES = {
    # Rates and yield curve
    "fed_funds_rate": {
        "id": "FEDFUNDS",
        "name": "Federal Funds Rate",
        "category": "rates",
    },
    "treasury_2y": {
        "id": "DGS2",
        "name": "2-Year Treasury Yield",
        "category": "rates",
    },
    "treasury_10y": {
        "id": "DGS10",
        "name": "10-Year Treasury Yield",
        "category": "rates",
    },
    "treasury_spread_10y_2y": {
        "id": "T10Y2Y",
        "name": "10Y-2Y Treasury Spread (Yield Curve)",
        "category": "rates",
    },
    # Inflation
    "cpi_yoy": {
        "id": "CPIAUCSL",
        "name": "CPI (All Urban Consumers)",
        "category": "inflation",
    },
    "core_pce": {
        "id": "PCEPILFE",
        "name": "Core PCE Price Index",
        "category": "inflation",
    },
    # Employment
    "unemployment_rate": {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "category": "employment",
    },
    "nonfarm_payrolls": {
        "id": "PAYEMS",
        "name": "Nonfarm Payrolls (Total)",
        "category": "employment",
    },
    "initial_claims": {
        "id": "ICSA",
        "name": "Initial Jobless Claims",
        "category": "employment",
    },
    # Growth
    "real_gdp": {
        "id": "GDPC1",
        "name": "Real GDP (Quarterly)",
        "category": "growth",
    },
    "industrial_production": {
        "id": "INDPRO",
        "name": "Industrial Production Index",
        "category": "growth",
    },
    # Credit and risk
    "high_yield_spread": {
        "id": "BAMLH0A0HYM2",
        "name": "ICE BofA US High Yield OAS",
        "category": "credit",
    },
    "vix": {
        "id": "VIXCLS",
        "name": "CBOE VIX (Volatility Index)",
        "category": "volatility",
    },
    # Consumer
    "consumer_sentiment": {
        "id": "UMCSENT",
        "name": "U of Michigan Consumer Sentiment",
        "category": "consumer",
    },
    "retail_sales": {
        "id": "RSXFS",
        "name": "Retail Sales (ex Food Services)",
        "category": "consumer",
    },
}


# ---------------------------------------------------------------------------
# Tool: macro environment snapshot
# ---------------------------------------------------------------------------

@tool
def get_macro_environment() -> dict:
    """
    Comprehensive macro environment snapshot from FRED.

    Returns the latest values and trends for key economic indicators
    across rates, inflation, employment, growth, credit, and volatility.

    This is CONTEXT, not a trading signal. Use it to understand the
    regime you're trading in:
    - Rising rates + falling yields curve = tightening → favor quality, avoid leveraged growth
    - Low VIX + tight credit spreads = complacency → watch for mean reversion
    - Rising claims + falling sentiment = slowing economy → defensive positioning
    - Falling rates + widening credit = easing → risk-on, favor growth

    The LLM's job is to synthesize these cross-domain signals into a
    coherent macro view that informs position-level decisions.
    """
    if not FRED_KEY:
        return {
            "error": "FRED_API_KEY not configured. Get a free key at "
                     "https://fred.stlouisfed.org/docs/api/api_key.html "
                     "and add it to your .env file.",
            "fallback_note": (
                "Without macro data, you're trading blind to the broader "
                "environment. At minimum, check the VIX and yield curve "
                "before making directional bets."
            ),
        }

    categories: dict[str, list] = {}

    for key, series_def in _MACRO_SERIES.items():
        data = _get_series_with_change(series_def["id"])
        category = series_def["category"]
        categories.setdefault(category, []).append({
            "indicator": series_def["name"],
            "series_id": series_def["id"],
            **data,
        })

    # Yield curve interpretation
    curve_data = _get_latest_value("T10Y2Y")
    curve_val = curve_data.get("value")
    yield_curve_regime = "unknown"
    if curve_val is not None:
        if curve_val < -0.5:
            yield_curve_regime = "deeply_inverted_recession_signal"
        elif curve_val < 0:
            yield_curve_regime = "inverted_caution"
        elif curve_val < 0.5:
            yield_curve_regime = "flat_transitional"
        elif curve_val < 1.5:
            yield_curve_regime = "normal_steepening"
        else:
            yield_curve_regime = "steep_growth_expected"

    # VIX regime
    vix_data = _get_latest_value("VIXCLS")
    vix_val = vix_data.get("value")
    vix_regime = "unknown"
    if vix_val is not None:
        if vix_val < 12:
            vix_regime = "extreme_complacency"
        elif vix_val < 18:
            vix_regime = "low_volatility"
        elif vix_val < 25:
            vix_regime = "moderate_volatility"
        elif vix_val < 35:
            vix_regime = "elevated_fear"
        else:
            vix_regime = "crisis_level_volatility"

    result = {
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "data_source": "FRED (Federal Reserve Economic Data)",
        "indicators_by_category": categories,
        "regime_assessment": {
            "yield_curve": yield_curve_regime,
            "yield_curve_spread": curve_val,
            "vix_regime": vix_regime,
            "vix_level": vix_val,
        },
        "synthesis_guidance": (
            "CROSS-REFERENCE these macro signals with your stock-level analysis. "
            "A bullish technical setup in a single stock means less if the macro "
            "environment is deteriorating (rising VIX, inverting curve, widening "
            "credit spreads). Conversely, a mediocre setup in a strong macro "
            "environment has better odds. The LLM's unique value is holding ALL "
            "of these signals in context simultaneously — something a human "
            "trader can do but much more slowly."
        ),
    }

    log_agent("macro_environment", {
        "yield_curve_regime": yield_curve_regime,
        "vix_regime": vix_regime,
    })

    return result
