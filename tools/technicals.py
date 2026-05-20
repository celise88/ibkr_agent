"""
Technical analysis tool.

Fetches historical bars from IBKR and computes a full indicator suite
deterministically in Python. Returns a structured summary — the LLM
synthesizes and interprets, but does NOT compute.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import ta
from ib_insync import util as ib_util
from langchain_core.tools import tool

from ibkr_agent.audit import log_agent
from ibkr_agent.connection import get_connection, ensure_connected
from ibkr_agent.tools._helpers import qualify_us_equity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator computation (all deterministic Python — no LLM involvement)
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a full suite of technical indicators to an OHLCV DataFrame.
    Operates in-place for efficiency; returns the same DataFrame.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # Trend — EMAs at multiple horizons + SMA-200
    df["ema_9"] = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)
    n = len(df)
    df["sma_200"] = ta.trend.sma_indicator(close, window=min(200, n)) if n >= 50 else pd.NA

    # MACD (12/26/9)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # ADX — trend strength
    df["adx_14"] = ta.trend.adx(high, low, close, window=14) if n >= 28 else pd.NA

    # Momentum
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["stoch_k"] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
    df["stoch_d"] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["vwap_approx"] = (
        (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    )

    return df


def _period_return(df: pd.DataFrame, periods: int) -> float | str:
    """Compute simple return over N periods."""
    if len(df) < periods + 1:
        return "insufficient_data"
    start = float(df.iloc[-1 - periods]["close"])
    end = float(df.iloc[-1]["close"])
    if start == 0:
        return "insufficient_data"
    return round((end / start - 1) * 100, 2)


def _classify_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "overbought"
    if rsi >= 60:
        return "bullish"
    if rsi <= 30:
        return "oversold"
    if rsi <= 40:
        return "bearish"
    return "neutral"


def _classify_adx(adx: float | None) -> str:
    if adx is None or pd.isna(adx):
        return "insufficient_data"
    if adx >= 40:
        return "strong_trend"
    if adx >= 25:
        return "trending"
    if adx >= 20:
        return "weak_trend"
    return "no_trend"


def _bb_position_pct(row: pd.Series) -> float | str:
    """Where price sits within Bollinger Bands (0% = lower, 100% = upper)."""
    try:
        upper = float(row["bb_upper"])
        lower = float(row["bb_lower"])
        close = float(row["close"])
        band_width = upper - lower
        if band_width <= 0:
            return 50.0
        return round((close - lower) / band_width * 100, 1)
    except (ValueError, TypeError):
        return "unavailable"


def _support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Identify naive support/resistance from recent highs and lows."""
    recent = df.tail(lookback)
    return {
        "recent_high": round(float(recent["high"].max()), 2),
        "recent_low": round(float(recent["low"].min()), 2),
        "recent_high_date": str(recent.loc[recent["high"].idxmax()].get("date", "unknown")),
        "recent_low_date": str(recent.loc[recent["low"].idxmin()].get("date", "unknown")),
    }


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
@ensure_connected
def get_technical_summary(symbol: str, lookback_days: int = 90) -> dict:
    """
    Compute a pre-digested technical summary for a US equity.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "NVDA").
        lookback_days: Calendar days of history to fetch (default 90, max 365).

    Returns:
        Structured dict with price action, trend, momentum, volatility,
        volume, and support/resistance sections. All indicators are
        pre-computed — do NOT attempt to recompute from raw data.
    """
    lookback_days = min(max(lookback_days, 30), 365)

    contract = qualify_us_equity(symbol)
    if contract is None:
        return {"error": f"Could not resolve contract for '{symbol}'."}

    ib = get_connection()

    # ------------------------------------------------------------------
    # Historical bars from IBKR
    # ------------------------------------------------------------------
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",                     # Empty = now
        durationStr=f"{lookback_days} D",
        barSizeSetting="1 day",
        whatToShow="ADJUSTED_LAST",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )

    if not bars or len(bars) < 10:
        return {
            "error": f"Insufficient historical data for {symbol} "
                     f"({len(bars) if bars else 0} bars returned).",
        }

    df = ib_util.df(bars)
    df.columns = [c.lower() for c in df.columns]
    df = _compute_indicators(df)

    latest = df.iloc[-1]
    close = float(latest["close"])
    prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else close

    # Volume profile
    vol_20d = df["volume"].tail(20).mean()
    vol_ratio = float(latest["volume"]) / vol_20d if vol_20d > 0 else 1.0

    # MACD histogram momentum
    macd_hist_expanding = False
    if len(df) > 1:
        curr_hist = abs(float(latest.get("macd_histogram", 0)))
        prev_hist = abs(float(df.iloc[-2].get("macd_histogram", 0)))
        macd_hist_expanding = curr_hist > prev_hist

    result = {
        "symbol": symbol.upper(),
        "as_of": str(latest.get("date", "unknown")),
        "bar_count": len(df),
        "price": {
            "latest_close": round(close, 2),
            "prev_close": round(prev_close, 2),
            "1d_change_pct": round((close / prev_close - 1) * 100, 2) if prev_close else 0,
            "5d_return_pct": _period_return(df, 5),
            "10d_return_pct": _period_return(df, 10),
            "20d_return_pct": _period_return(df, 20),
            "above_ema_50": close > float(latest["ema_50"]) if pd.notna(latest["ema_50"]) else "insufficient_data",
            "above_sma_200": (
                close > float(latest["sma_200"])
                if pd.notna(latest.get("sma_200")) else "insufficient_data"
            ),
        },
        "trend": {
            "ema_9_vs_21": "bullish" if latest["ema_9"] > latest["ema_21"] else "bearish",
            "ema_21_vs_50": "bullish" if latest["ema_21"] > latest["ema_50"] else "bearish",
            "macd_crossover": (
                "bullish" if latest["macd_line"] > latest["macd_signal"] else "bearish"
            ),
            "macd_histogram_direction": "expanding" if macd_hist_expanding else "contracting",
            "adx_14": round(float(latest["adx_14"]), 1) if pd.notna(latest.get("adx_14")) else "insufficient_data",
            "adx_interpretation": _classify_adx(
                float(latest["adx_14"]) if pd.notna(latest.get("adx_14")) else None
            ),
        },
        "momentum": {
            "rsi_14": round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else "unavailable",
            "rsi_zone": _classify_rsi(float(latest["rsi_14"])) if pd.notna(latest["rsi_14"]) else "unavailable",
            "stoch_k": round(float(latest["stoch_k"]), 1) if pd.notna(latest.get("stoch_k")) else "unavailable",
            "stoch_d": round(float(latest["stoch_d"]), 1) if pd.notna(latest.get("stoch_d")) else "unavailable",
        },
        "volatility": {
            "atr_14": round(float(latest["atr_14"]), 2) if pd.notna(latest["atr_14"]) else "unavailable",
            "atr_pct_of_price": (
                round(float(latest["atr_14"]) / close * 100, 2)
                if pd.notna(latest["atr_14"]) and close > 0 else "unavailable"
            ),
            "bb_position_pct": _bb_position_pct(latest),
            "bb_width_pct": (
                round(float(latest["bb_width"]) * 100, 2)
                if pd.notna(latest.get("bb_width")) else "unavailable"
            ),
        },
        "volume": {
            "latest_volume": int(latest["volume"]),
            "20d_avg_volume": int(vol_20d) if vol_20d > 0 else "unavailable",
            "vs_20d_avg_ratio": round(vol_ratio, 2),
            "conviction": (
                "high" if vol_ratio > 1.5
                else "low" if vol_ratio < 0.5
                else "normal"
            ),
        },
        "levels": _support_resistance(df, lookback=20),
    }

    log_agent("technical_analysis", {
        "symbol": symbol.upper(),
        "close": close,
        "rsi": result["momentum"]["rsi_14"],
        "trend_ema": result["trend"]["ema_9_vs_21"],
        "macd": result["trend"]["macd_crossover"],
    })

    return result
