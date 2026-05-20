"""
Earnings call transcript tool.

This is the single highest-value tool for information synthesis. An LLM
reading a full earnings transcript can detect management tone shifts,
hedging language, confidence changes, and guidance nuance that no
indicator or summary can capture.

Two data sources, tried in order:
  1. SEC EDGAR 8-K exhibits (free, no key, raw transcript text)
  2. API Ninjas (free tier, structured transcript + sentiment + guidance)

The EDGAR path gives raw text the LLM synthesizes from scratch.
The API Ninjas path gives pre-structured data with sentiment scores
as a cross-reference. Both are valuable; the combination is best.

Requires: API_NINJAS_KEY in environment (free at https://api-ninjas.com)
          SEC EDGAR access needs no key.
"""

from __future__ import annotations

import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from langchain_core.tools import tool

from ibkr_agent.audit import log_agent

logger = logging.getLogger(__name__)

API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY", "")
_EDGAR_HEADERS = {"User-Agent": "IBKRAgent/0.1 (trading-agent@example.com)"}
_EDGAR_BASE = "https://data.sec.gov"

# Maximum transcript length to return (tokens are expensive; 15k chars ≈ 4k tokens)
_MAX_TRANSCRIPT_CHARS = 15_000


# ---------------------------------------------------------------------------
# HTML stripping utility
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text extractor."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._skip = False
        if tag in {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.parts)
        # Collapse whitespace runs but preserve paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _strip_html(html: str) -> str:
    """Convert HTML to readable plain text."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# Source 1: SEC EDGAR 8-K transcript exhibits
# ---------------------------------------------------------------------------

def _resolve_cik(ticker: str) -> str | None:
    """Resolve ticker to 10-digit zero-padded CIK."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        req = Request(url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("CIK resolution failed for %s: %s", ticker, exc)
    return None


def _fetch_edgar_transcript(ticker: str) -> dict[str, Any] | None:
    """
    Attempt to find an earnings call transcript in recent 8-K filings.

    Strategy: Pull recent 8-K filings, look for exhibits labeled as
    "earnings" or "transcript" (typically Exhibit 99.1), fetch the
    exhibit HTML, and strip it to plain text.
    """
    cik = _resolve_cik(ticker)
    if cik is None:
        return None

    # Get recent filings
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        req = Request(url, headers=_EDGAR_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("EDGAR submissions failed for %s: %s", ticker, exc)
        return None

    company_name = data.get("name", ticker)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    # Find recent 8-K filings that look like earnings transcript exhibits
    transcript_keywords = {
        "transcript", "earnings call", "conference call",
        "earnings release", "press release",
    }

    for i in range(min(len(forms), 40)):
        if forms[i] not in {"8-K", "8-K/A"}:
            continue

        # Check the filing index for transcript-like exhibits
        accession_clean = accessions[i].replace("-", "")
        cik_stripped = cik.lstrip("0")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_stripped}/{accession_clean}/"
        )

        try:
            # Fetch the filing index to find exhibit documents
            filing_index_url = f"{_EDGAR_BASE}/Archives/edgar/data/{cik_stripped}/{accession_clean}/index.json"
            req = Request(filing_index_url, headers=_EDGAR_HEADERS)
            with urlopen(req, timeout=10) as resp:
                index_data = json.loads(resp.read().decode("utf-8"))
        except (URLError, json.JSONDecodeError):
            continue

        # Look through the filing's documents for transcript exhibits
        items = index_data.get("directory", {}).get("item", [])
        for item in items:
            doc_name = item.get("name", "").lower()
            # Typical transcript exhibit patterns: ex99-1.htm, exhibit991.htm, etc.
            # Also check for press releases which sometimes contain call transcripts
            is_exhibit = any(
                pattern in doc_name
                for pattern in ["ex99", "exhibit99", "ex-99", "transcript", "earningscall"]
            )
            if not is_exhibit:
                continue
            if not doc_name.endswith((".htm", ".html", ".txt")):
                continue

            # Fetch this exhibit
            exhibit_url = f"{index_url}{item['name']}"
            try:
                req = Request(exhibit_url, headers=_EDGAR_HEADERS)
                with urlopen(req, timeout=15) as resp:
                    content = resp.read().decode("utf-8", errors="replace")
            except (URLError, UnicodeDecodeError):
                continue

            # Extract text and check if it looks like a transcript
            text = _strip_html(content) if "<html" in content.lower() or "<p" in content.lower() else content

            # Heuristic: a real transcript has dialogue markers and is substantial
            has_dialogue = any(
                marker in text.lower()
                for marker in [
                    "operator:", "good morning", "good afternoon",
                    "question-and-answer", "q&a session",
                    "opening remarks", "prepared remarks",
                    "conference call", "earnings call",
                ]
            )

            if has_dialogue and len(text) > 2000:
                return {
                    "source": "sec_edgar",
                    "company": company_name,
                    "filing_date": dates[i],
                    "accession": accessions[i],
                    "exhibit_url": exhibit_url,
                    "transcript_text": text[:_MAX_TRANSCRIPT_CHARS],
                    "full_length_chars": len(text),
                    "truncated": len(text) > _MAX_TRANSCRIPT_CHARS,
                }

    return None


# ---------------------------------------------------------------------------
# Source 2: API Ninjas earnings transcript
# ---------------------------------------------------------------------------

def _fetch_api_ninjas_transcript(
    ticker: str,
    year: int | None = None,
    quarter: int | None = None,
) -> dict[str, Any] | None:
    """
    Fetch an earnings transcript from API Ninjas.
    Returns structured data with transcript text, sentiment, guidance, and risks.
    """
    if not API_NINJAS_KEY:
        return None

    params = {"ticker": ticker.upper()}
    if year is not None and quarter is not None:
        params["year"] = str(year)
        params["quarter"] = str(quarter)

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://api.api-ninjas.com/v1/earningstranscript?{query}"

    try:
        req = Request(url, headers={
            "X-Api-Key": API_NINJAS_KEY,
            "User-Agent": "IBKRAgent/0.1",
        })
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        logger.debug("API Ninjas transcript failed for %s: %s", ticker, exc)
        return None

    if not data or "transcript" not in data:
        return None

    transcript_text = data.get("transcript", "")

    return {
        "source": "api_ninjas",
        "ticker": data.get("ticker", ticker),
        "date": data.get("date"),
        "year": data.get("year"),
        "quarter": data.get("quarter"),
        "earnings_timing": data.get("earnings_timing"),
        "transcript_text": transcript_text[:_MAX_TRANSCRIPT_CHARS],
        "full_length_chars": len(transcript_text),
        "truncated": len(transcript_text) > _MAX_TRANSCRIPT_CHARS,
        "participants": data.get("participants", []),
        "summary": data.get("summary", ""),
        "guidance": data.get("guidance", ""),
        "risk_factors": data.get("risk_factors", ""),
        "overall_sentiment": data.get("overall_sentiment"),
        "sentiment_rationale": data.get("overall_sentiment_rationale", ""),
    }


def _fetch_api_ninjas_available(ticker: str) -> list[dict[str, str]]:
    """List available transcript quarters for a ticker."""
    if not API_NINJAS_KEY:
        return []

    url = f"https://api.api-ninjas.com/v1/earningstranscriptsearch?ticker={ticker.upper()}"
    try:
        req = Request(url, headers={
            "X-Api-Key": API_NINJAS_KEY,
            "User-Agent": "IBKRAgent/0.1",
        })
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, list) else []
    except (URLError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Tool: get earnings transcript
# ---------------------------------------------------------------------------

@tool
def get_earnings_transcript(
    symbol: str,
    year: int | None = None,
    quarter: int | None = None,
) -> dict:
    """
    Fetch the earnings call transcript for a company.

    This is the MOST VALUABLE tool for information synthesis. A full
    transcript lets you detect:
    - Management confidence shifts (hedging language, qualifiers)
    - Guidance tone vs actual numbers (sandbagging vs over-promising)
    - Analyst questions that signal market concerns
    - Specific forward-looking language about demand, pipeline, margins
    - Changes in how management discusses risks vs prior quarters

    Tries SEC EDGAR first (raw transcript from 8-K exhibits), then
    falls back to API Ninjas (structured transcript + pre-computed
    sentiment and guidance extraction).

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "NVDA").
        year: Specific year (e.g., 2026). If omitted, returns most recent.
        quarter: Specific quarter (1-4). If omitted, returns most recent.

    Returns:
        Dict with transcript text, metadata, and (from API Ninjas)
        pre-computed sentiment, guidance, and risk factor summaries.
    """
    symbol = symbol.upper().strip()

    edgar_result = None
    ninjas_result = None

    # --- Source 1: SEC EDGAR ---
    logger.info("Searching EDGAR for %s earnings transcript...", symbol)
    edgar_result = _fetch_edgar_transcript(symbol)
    if edgar_result:
        logger.info(
            "Found EDGAR transcript for %s (filed %s, %d chars)",
            symbol, edgar_result["filing_date"], edgar_result["full_length_chars"],
        )

    # --- Source 2: API Ninjas ---
    logger.info("Fetching API Ninjas transcript for %s...", symbol)
    ninjas_result = _fetch_api_ninjas_transcript(symbol, year=year, quarter=quarter)
    if ninjas_result:
        logger.info(
            "Found API Ninjas transcript for %s Q%s %s (%d chars, sentiment=%.2f)",
            symbol,
            ninjas_result.get("quarter", "?"),
            ninjas_result.get("year", "?"),
            ninjas_result.get("full_length_chars", 0),
            ninjas_result.get("overall_sentiment", 0) or 0,
        )

    # --- Assemble result ---
    if not edgar_result and not ninjas_result:
        # List what's available so the agent can retry with specific quarter
        available = _fetch_api_ninjas_available(symbol)
        return {
            "symbol": symbol,
            "status": "not_found",
            "error": (
                f"No earnings transcript found for {symbol}. "
                "The company may not file transcripts on EDGAR, or the "
                "API Ninjas free tier may not cover this ticker."
            ),
            "available_quarters": available[:8] if available else [],
            "suggestion": (
                "Try specifying a year and quarter from the available list, "
                "or use get_sec_filings to check for 8-K press releases "
                "which may contain earnings commentary."
            ),
        }

    result: dict[str, Any] = {
        "symbol": symbol,
        "status": "found",
        "sources": [],
    }

    # Prefer API Ninjas for structured data (sentiment, guidance, risk)
    if ninjas_result:
        result["sources"].append("api_ninjas")
        result["structured"] = {
            "date": ninjas_result["date"],
            "year": ninjas_result["year"],
            "quarter": ninjas_result["quarter"],
            "earnings_timing": ninjas_result["earnings_timing"],
            "participants": ninjas_result["participants"][:10],  # Cap for context
            "summary": ninjas_result["summary"],
            "guidance": ninjas_result["guidance"],
            "risk_factors": ninjas_result["risk_factors"],
            "overall_sentiment": ninjas_result["overall_sentiment"],
            "sentiment_rationale": ninjas_result["sentiment_rationale"],
        }

    # Include raw transcript text (prefer EDGAR if available, otherwise Ninjas)
    if edgar_result:
        result["sources"].append("sec_edgar")
        result["transcript"] = {
            "text": edgar_result["transcript_text"],
            "source": "sec_edgar",
            "filing_date": edgar_result["filing_date"],
            "exhibit_url": edgar_result["exhibit_url"],
            "full_length_chars": edgar_result["full_length_chars"],
            "truncated": edgar_result["truncated"],
        }
    elif ninjas_result:
        result["transcript"] = {
            "text": ninjas_result["transcript_text"],
            "source": "api_ninjas",
            "full_length_chars": ninjas_result["full_length_chars"],
            "truncated": ninjas_result["truncated"],
        }

    result["synthesis_instructions"] = (
        "READ THIS TRANSCRIPT CAREFULLY. Your edge is in synthesis:\n"
        "1. TONE ANALYSIS: Is management confident or hedging? Count "
        "qualifiers ('we believe', 'we expect' vs 'we will', 'we are').\n"
        "2. GUIDANCE vs REALITY: Compare the guidance section to the "
        "actual numbers. Are they sandbagging (guiding low after a beat) "
        "or over-promising?\n"
        "3. ANALYST CONCERNS: What are analysts asking about? Their "
        "questions reveal what the market is worried about.\n"
        "4. COMPARE TO PRIOR: If you've seen prior transcripts, flag "
        "any language that changed — new risks mentioned, old risks "
        "dropped, changed emphasis on segments.\n"
        "5. CROSS-REFERENCE: Compare what management says here against "
        "the SEC filing numbers and technical price action. Divergence "
        "between narrative and data is a strong signal."
    )

    log_agent("earnings_transcript", {
        "symbol": symbol,
        "sources": result["sources"],
        "transcript_length": result.get("transcript", {}).get("full_length_chars", 0),
        "sentiment": result.get("structured", {}).get("overall_sentiment"),
    })

    return result


@tool
def list_available_transcripts(symbol: str) -> dict:
    """
    List all available earnings transcript quarters for a symbol.

    Use this to see which quarters have transcripts before requesting
    a specific one. Useful for comparative analysis across quarters.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
    """
    symbol = symbol.upper().strip()
    available = _fetch_api_ninjas_available(symbol)

    return {
        "symbol": symbol,
        "available_transcripts": available[:20],
        "total_available": len(available),
        "data_source": "api_ninjas",
        "note": (
            "Request specific quarters with get_earnings_transcript(symbol, year, quarter) "
            "to compare management tone and guidance across periods. Quarter-over-quarter "
            "changes in language and emphasis are often more revealing than the numbers."
        ),
    }
