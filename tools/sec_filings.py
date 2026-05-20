"""
SEC filings tool — the LLM's primary information-synthesis edge.

Uses the free SEC EDGAR API (no key required) and edgartools to pull
recent filings and extract the textual sections that matter most for
investment decisions:

  - 10-K/10-Q: risk factors, MD&A, financial highlights
  - 8-K: material events (earnings results, leadership changes, M&A)
  - DEF 14A: executive compensation, shareholder proposals

The LLM's job is to READ these and synthesize — not summarize bullet points,
but identify discrepancies between management tone and numbers, compare
current language to prior quarters, and flag what's genuinely new.

Requires: pip install edgartools
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from langchain_core.tools import tool

from ibkr_agent.audit import log_agent

logger = logging.getLogger(__name__)

# EDGAR requires a User-Agent with contact info
_EDGAR_HEADERS = {"User-Agent": "IBKRAgent/0.1 (trading-agent@example.com)"}
_EDGAR_BASE = "https://data.sec.gov"
_EFTS_BASE = "https://efts.sec.gov/LATEST"


# ---------------------------------------------------------------------------
# CIK resolution (ticker → CIK mapping)
# ---------------------------------------------------------------------------

_CIK_CACHE: dict[str, str] = {}


def _resolve_cik(ticker: str) -> str | None:
    """Resolve a ticker to a 10-digit zero-padded CIK via EDGAR."""
    ticker = ticker.upper().strip()
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    url = f"{_EDGAR_BASE}/submissions/CIK{ticker}.json"
    # EDGAR also supports a ticker→CIK lookup file
    lookup_url = "https://www.sec.gov/files/company_tickers.json"

    try:
        req = Request(lookup_url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                return cik

    except (URLError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("CIK lookup failed for %s: %s", ticker, exc)

    return None


# ---------------------------------------------------------------------------
# EDGAR submissions API (free, structured)
# ---------------------------------------------------------------------------

def _fetch_recent_filings(
    cik: str,
    form_types: set[str] | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch recent filing metadata from EDGAR submissions endpoint.
    Returns a list of dicts with filing type, date, description, and URLs.
    """
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        req = Request(url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.warning("EDGAR submissions request failed for CIK %s: %s", cik, exc)
        return []

    company_name = data.get("name", "Unknown")
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    filings = []
    for i in range(min(len(forms), 100)):  # EDGAR returns up to 100
        form = forms[i]
        if form_types and form not in form_types:
            continue

        accession_clean = accessions[i].replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}"
        )

        filings.append({
            "company": company_name,
            "form_type": form,
            "filing_date": dates[i],
            "accession_number": accessions[i],
            "description": descriptions[i] if i < len(descriptions) else "",
            "url": filing_url,
        })

        if len(filings) >= max_results:
            break

    return filings


# ---------------------------------------------------------------------------
# XBRL financial facts (structured financials, free)
# ---------------------------------------------------------------------------

def _fetch_financial_facts(cik: str, metrics: list[str] | None = None) -> dict[str, Any]:
    """
    Pull structured financial data from EDGAR's XBRL companyfacts endpoint.
    Returns the most recent values for key financial metrics.
    """
    if metrics is None:
        metrics = [
            "Revenues",
            "NetIncomeLoss",
            "EarningsPerShareDiluted",
            "OperatingIncomeLoss",
            "GrossProfit",
            "Assets",
            "Liabilities",
            "StockholdersEquity",
            "CashAndCashEquivalentsAtCarryingValue",
            "LongTermDebt",
            "OperatingCashFlow",
        ]

    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        req = Request(url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("XBRL companyfacts failed for CIK %s: %s", cik, exc)
        return {}

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    results = {}

    for metric in metrics:
        if metric not in us_gaap:
            continue

        units = us_gaap[metric].get("units", {})
        # Most financial metrics are in USD or USD/shares
        values = units.get("USD", units.get("USD/shares", []))
        if not values:
            continue

        # Get the most recent 10-K and 10-Q values
        annual = [v for v in values if v.get("form") == "10-K"]
        quarterly = [v for v in values if v.get("form") == "10-Q"]

        latest_annual = annual[-1] if annual else None
        latest_quarterly = quarterly[-1] if quarterly else None
        prev_annual = annual[-2] if len(annual) > 1 else None

        entry = {}
        if latest_quarterly:
            entry["latest_quarterly"] = {
                "value": latest_quarterly["val"],
                "period": latest_quarterly.get("end", "unknown"),
                "filed": latest_quarterly.get("filed", "unknown"),
            }
        if latest_annual:
            entry["latest_annual"] = {
                "value": latest_annual["val"],
                "period": latest_annual.get("end", "unknown"),
                "filed": latest_annual.get("filed", "unknown"),
            }
        if prev_annual:
            entry["prior_annual"] = {
                "value": prev_annual["val"],
                "period": prev_annual.get("end", "unknown"),
            }
            if latest_annual and prev_annual["val"] != 0:
                yoy_change = (latest_annual["val"] / prev_annual["val"] - 1) * 100
                entry["yoy_change_pct"] = round(yoy_change, 2)

        if entry:
            results[metric] = entry

    return results


# ---------------------------------------------------------------------------
# Full-text search (EFTS — EDGAR Full-Text Search)
# ---------------------------------------------------------------------------

def _search_filings_fulltext(
    query: str,
    ticker: str | None = None,
    form_types: list[str] | None = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Full-text search across all EDGAR filings using the EFTS API.
    Useful for finding specific disclosures (e.g., "material weakness",
    "goodwill impairment", "going concern").
    """
    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        "enddt": datetime.now().strftime("%Y-%m-%d"),
    }
    if form_types:
        params["forms"] = ",".join(form_types)

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_EFTS_BASE}/search-index?{query_string}"

    try:
        req = Request(url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("EFTS search failed for '%s': %s", query, exc)
        return []

    results = []
    for hit in data.get("hits", {}).get("hits", [])[:max_results]:
        source = hit.get("_source", {})
        entities = source.get("entity_name", "")
        tickers = source.get("tickers", "")

        # Filter by ticker if specified
        if ticker and ticker.upper() not in str(tickers).upper():
            continue

        results.append({
            "entity": entities,
            "form_type": source.get("form_type", ""),
            "filing_date": source.get("file_date", ""),
            "description": source.get("display_names", [""])[0] if source.get("display_names") else "",
            "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('file_num', '')}",
        })

    return results


# ---------------------------------------------------------------------------
# Tool: comprehensive SEC filing analysis
# ---------------------------------------------------------------------------

@tool
def get_sec_filings(
    symbol: str,
    filing_types: str = "10-K,10-Q,8-K",
    max_filings: int = 8,
) -> dict:
    """
    Pull recent SEC filings and structured financial data for a company.

    This is a PRIMARY information source — use it to understand a company's
    actual financial position, recent material events, and management
    disclosures. Cross-reference this with market price action and news.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "NVDA").
        filing_types: Comma-separated form types (default: "10-K,10-Q,8-K").
        max_filings: Maximum number of filings to return (default 8).

    Returns:
        Structured dict with:
        - recent_filings: metadata for recent filings with URLs
        - financial_snapshot: key XBRL metrics (revenue, earnings, cash, debt)
        - yoy_trends: year-over-year changes for major line items
    """
    symbol = symbol.upper().strip()
    form_types = {ft.strip() for ft in filing_types.split(",")}

    # Resolve CIK
    cik = _resolve_cik(symbol)
    if cik is None:
        return {"error": f"Could not resolve SEC CIK for ticker '{symbol}'."}

    # Pull filing metadata
    filings = _fetch_recent_filings(cik, form_types=form_types, max_results=max_filings)

    # Pull structured financials
    financials = _fetch_financial_facts(cik)

    # Separate 8-K material events for emphasis
    material_events = [f for f in filings if f["form_type"] == "8-K"]
    periodic_reports = [f for f in filings if f["form_type"] in {"10-K", "10-Q"}]

    result = {
        "symbol": symbol,
        "cik": cik,
        "periodic_reports": periodic_reports,
        "material_events": material_events,
        "financial_snapshot": financials,
        "analysis_guidance": (
            "IMPORTANT: Your edge is in READING and SYNTHESIZING this data, not "
            "summarizing it. Look for: (1) YoY changes that diverge from market "
            "expectations, (2) new risk factors or removed risk factors vs prior "
            "filings, (3) changes in revenue mix or segment performance, "
            "(4) cash flow trends vs earnings trends (divergence = red flag), "
            "(5) recent 8-K filings that signal material events the market may "
            "not have fully priced in."
        ),
    }

    log_agent("sec_filings", {
        "symbol": symbol,
        "filing_count": len(filings),
        "has_financials": bool(financials),
    })

    return result


@tool
def search_sec_disclosures(
    query: str,
    symbol: str | None = None,
    form_types: str = "10-K,10-Q,8-K",
) -> dict:
    """
    Full-text search across SEC filings for specific disclosures.

    Use this to find specific risk language, material events, or disclosures
    across filings. Powerful for identifying companies with specific exposures.

    Example queries:
        - "material weakness" (accounting red flags)
        - "goodwill impairment" (write-down risk)
        - "going concern" (bankruptcy risk)
        - "cybersecurity incident" (breach disclosures)
        - "supply chain disruption" (operational risk)
        - "tariff" or "trade restriction" (policy exposure)

    Args:
        query: Search terms to find in filing text.
        symbol: Optional ticker to filter results.
        form_types: Comma-separated form types to search.
    """
    ft_list = [ft.strip() for ft in form_types.split(",")]

    results = _search_filings_fulltext(
        query=query,
        ticker=symbol,
        form_types=ft_list,
        max_results=10,
    )

    return {
        "query": query,
        "symbol_filter": symbol,
        "results_count": len(results),
        "filings": results,
        "usage_note": (
            "Review these filings for the specific language around your search "
            "term. Context matters — 'material weakness' in a remediation "
            "update is very different from a new disclosure."
        ),
    }
