"""
Agent tools — the LLM's interface to IBKR + information sources.

Two categories of tools, reflecting where LLMs add value:

INFORMATION EDGE (where LLMs beat humans):
  - Earnings transcripts: raw call text for tone/guidance/risk synthesis
  - SEC filings: read 10-Ks, 8-Ks, financial facts, full-text search
  - Earnings analysis: surprise history, analyst consensus, price targets
  - Macro environment: FRED economic data, regime classification
  - News/sentiment: headline aggregation and sentiment signals

EXECUTION (where deterministic code beats LLMs):
  - Portfolio snapshot: account state and position P&L
  - Technical summary: pre-computed indicators (LLM interprets, doesn't compute)
  - Trade execution: orders with hard-coded risk guardrails
"""

from ibkr_agent.tools.portfolio import get_portfolio_snapshot
from ibkr_agent.tools.technicals import get_technical_summary
from ibkr_agent.tools.execution import place_trade, place_bracket_trade, close_position
from ibkr_agent.tools.news import get_news_sentiment
from ibkr_agent.tools.sec_filings import get_sec_filings, search_sec_disclosures
from ibkr_agent.tools.earnings import get_earnings_analysis, get_earnings_calendar
from ibkr_agent.tools.macro import get_macro_environment
from ibkr_agent.tools.transcripts import get_earnings_transcript, list_available_transcripts

# Ordered by information value — the agent should reach for information
# tools BEFORE execution tools
ALL_TOOLS = [
    # Information synthesis (LLM's primary edge)
    get_earnings_transcript,
    list_available_transcripts,
    get_sec_filings,
    search_sec_disclosures,
    get_earnings_analysis,
    get_earnings_calendar,
    get_macro_environment,
    get_news_sentiment,
    # Market data (pre-computed, LLM interprets)
    get_portfolio_snapshot,
    get_technical_summary,
    # Execution (LLM decides, code enforces)
    place_trade,
    place_bracket_trade,
    close_position,
]

__all__ = [
    "get_portfolio_snapshot",
    "get_technical_summary",
    "place_trade",
    "place_bracket_trade",
    "close_position",
    "get_news_sentiment",
    "get_sec_filings",
    "search_sec_disclosures",
    "get_earnings_analysis",
    "get_earnings_calendar",
    "get_macro_environment",
    "get_earnings_transcript",
    "list_available_transcripts",
    "ALL_TOOLS",
]
