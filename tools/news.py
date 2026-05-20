"""
News and sentiment tool.

Provides two data sources:
  1. IBKR's built-in news feed (free with any account, limited coverage)
  2. Polygon.io REST API (optional, requires POLYGON_API_KEY)

If neither is available, returns a structured "no data" response so the
agent can proceed with technicals alone rather than erroring out.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from langchain_core.tools import tool

from ibkr_agent.audit import log_agent
from ibkr_agent.connection import get_connection, ensure_connected
from ibkr_agent.tools._helpers import qualify_us_equity

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")


# ---------------------------------------------------------------------------
# IBKR news (always available, limited depth)
# ---------------------------------------------------------------------------

def _fetch_ibkr_news(symbol: str, max_items: int = 10) -> list[dict[str, Any]]:
    """
    Fetch recent news headlines from IBKR's news feed for a contract.
    IBKR provides headlines from subscribed providers (BZ, FLY, DJ, etc.).
    Coverage varies by subscription.
    """
    contract = qualify_us_equity(symbol)
    if contract is None:
        return []

    ib = get_connection()
    try:
        headlines = ib.reqHistoricalNews(
            conId=contract.conId,
            providerCodes="BZ+FLY+DJ+MT+BRFG",  # Common free/included providers
            startDateTime="",
            endDateTime="",
            totalResults=max_items,
        )
        ib.sleep(1)

        results = []
        for h in headlines:
            results.append({
                "source": h.providerCode,
                "timestamp": str(h.time),
                "headline": h.headline,
            })
        return results

    except Exception as exc:
        logger.debug("IBKR news request failed for %s: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Polygon.io news (optional, richer content)
# ---------------------------------------------------------------------------

def _fetch_polygon_news(
    symbol: str, lookback_hours: int = 48, max_items: int = 10
) -> list[dict[str, Any]]:
    """
    Fetch news articles from Polygon.io's REST API.
    Requires POLYGON_API_KEY environment variable.
    Returns enriched articles with sentiment if available.
    """
    if not POLYGON_API_KEY:
        return []

    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?ticker={symbol.upper()}"
        f"&published_utc.gte={published_after}"
        f"&limit={max_items}"
        f"&sort=published_utc"
        f"&order=desc"
        f"&apiKey={POLYGON_API_KEY}"
    )

    try:
        req = Request(url, headers={"User-Agent": "ibkr-agent/0.1"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for article in data.get("results", []):
            # Polygon provides sentiment via `insights` on paid tiers
            insights = article.get("insights", [])
            ticker_sentiment = None
            for insight in insights:
                if insight.get("ticker", "").upper() == symbol.upper():
                    ticker_sentiment = insight.get("sentiment")
                    break

            results.append({
                "source": article.get("publisher", {}).get("name", "unknown"),
                "timestamp": article.get("published_utc", "unknown"),
                "title": article.get("title", ""),
                "description": (article.get("description", "") or "")[:300],
                "sentiment": ticker_sentiment,
                "url": article.get("article_url", ""),
            })

        return results

    except (URLError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("Polygon news request failed for %s: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
@ensure_connected
def get_news_sentiment(symbol: str, lookback_hours: int = 48) -> dict:
    """
    Fetch recent news and sentiment data for a symbol.

    Pulls from IBKR's built-in news feed and (if configured) Polygon.io.
    Use this alongside technical analysis to form a more complete thesis.
    Do NOT trade on news alone — always cross-reference with technicals.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        lookback_hours: How far back to search for news (default 48 hours).

    Returns:
        Structured dict with headline lists from available sources and
        a basic sentiment summary.
    """
    symbol = symbol.upper().strip()
    lookback_hours = min(max(lookback_hours, 1), 168)  # Cap at 1 week

    ibkr_news = _fetch_ibkr_news(symbol, max_items=10)
    polygon_news = _fetch_polygon_news(symbol, lookback_hours=lookback_hours, max_items=10)

    # Deduplicate by approximate title matching
    seen_titles = set()
    combined = []
    for article in polygon_news + ibkr_news:
        title = (article.get("title") or article.get("headline", "")).lower().strip()
        # Crude dedup: first 60 chars of title
        key = title[:60]
        if key and key not in seen_titles:
            seen_titles.add(key)
            combined.append(article)

    # Basic sentiment tally from Polygon (if available)
    sentiments = [a["sentiment"] for a in polygon_news if a.get("sentiment")]
    sentiment_summary = {
        "positive": sum(1 for s in sentiments if s == "positive"),
        "negative": sum(1 for s in sentiments if s == "negative"),
        "neutral": sum(1 for s in sentiments if s == "neutral"),
    }

    data_sources = []
    if ibkr_news:
        data_sources.append("ibkr")
    if polygon_news:
        data_sources.append("polygon")

    result = {
        "symbol": symbol,
        "lookback_hours": lookback_hours,
        "data_sources": data_sources,
        "total_articles": len(combined),
        "articles": combined[:15],  # Cap context size
        "sentiment_summary": sentiment_summary if sentiments else "no_sentiment_data_available",
        "note": (
            "News sentiment is supplementary — always cross-reference with "
            "technical indicators before making trade decisions."
        ),
    }

    if not combined:
        result["warning"] = (
            "No recent news found. This could mean the symbol has low "
            "coverage, or news providers are unavailable. Proceed with "
            "technical analysis only."
        )

    log_agent("news_sentiment", {
        "symbol": symbol,
        "article_count": len(combined),
        "sources": data_sources,
    })

    return result
